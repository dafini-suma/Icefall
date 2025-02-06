#!/bin/bash

set -eou pipefail

if grep -q docker /proc/1/cgroup; then 
   scripts=/scripts
   local=/local
else
   scripts=./scripts
   local=./local
fi

config_path="${scripts}/config.json"

cuda_visible_devices=$(cat $config_path | jq  ."cuda_visible_devices")
export CUDA_VISIBLE_DEVICES=$cuda_visible_devices
# ------------------------- Begin Configuration -------------------------
language=$(cat $config_path | jq -r ."language")

kaldi_train_dir=$(cat $config_path | jq  ."kaldi_train_dir"|sed 's/"//g')
kaldi_val_dir=$(cat $config_path | jq  ."kaldi_val_dir"|sed 's/"//g')
kaldi_test_dirs=$(cat $config_path | jq  ."kaldi_test_dirs"|sed 's/"//g')
musan_download_dir=$(cat $config_path | jq  ."musan_download_dir"|sed 's/"//g')

on_the_fly_feats=$(cat $config_path | jq -r ."on_the_fly_feats")
bpe_vocab_size=$(cat $config_path | jq -r ."bpe_vocab_size")
sampling_rate=$(cat $config_path | jq -r ."sampling_rate")
feature_dim=$(cat $config_path | jq -r ."feature_dim")
num_jobs=$(cat $config_path | jq -r ."num_jobs")
prepare_dev_from_train=$(cat $config_path | jq -r ."prepare_dev_from_train")
dev_from_train_percent=$(cat $config_path | jq -r ."dev_from_train_percent")

lang_dir=data/${language}/lang_bpe_${bpe_vocab_size}
manifest_dir="data/${language}/manifests"
cut_dir="data/${language}/manifests"
lm_dir="data/${language}/lm"

stage=$(cat $config_path | jq -r ."stage")
stop_stage=$(cat $config_path | jq -r ."stop_stage")
kaldi_root_dir=$(cat $config_path | jq -r ."kaldi_root_dir")
icefall_root_dir=$(cat $config_path | jq -r ."icefall_root_dir")
# ------------------------- End Configuration -------------------------

shared=$icefall_root_dir/icefall/shared
export PYTHONPATH=$icefall_root_dir

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    if [ ! -d $musan_download_dir/musan ]; then
        log "Stage 0: Downloading MUSAN data."
        lhotse download musan $musan_download_dir
    fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Preparing icefall manifests."
    mkdir -p $manifest_dir
    for data in $kaldi_test_dirs $kaldi_val_dir $kaldi_train_dir; do
        bdata=$(basename $data)
        log "Processing $bdata and $data "
        python3 $scripts/data_parser_kaldi.py \
            -i $data \
            -o $manifest_dir/$bdata \
            -s $sampling_rate \
            -k $kaldi_root_dir
    done
# '''
#     log "Processing musan"
#     # We assume that you have downloaded the musan corpus
#     # to data/musan
#     if [ ! -e $manifest_dir/.musan.done ]; then
#         lhotse prepare musan $musan_download_dir/musan $manifest_dir/musan
#     fi
# '''
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Preparing test cuts."
    for data in $kaldi_test_dirs $kaldi_val_dir; do
        bdata=$(basename $data)
        log "Processing $bdata"
        python3 $scripts/prepare_cuts.py \
            -c "${language}_${bdata}_cuts" \
            -d $manifest_dir/$bdata \
            -s $sampling_rate \
            -o $cut_dir \
            -v
        log "Succesfully stored $cut_dir/${language}_${bdata}_cuts.jsonl.gz"
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Preparing training cuts."
    data=$kaldi_train_dir
    bdata=$(basename $data)
    python3 $scripts/prepare_cuts.py \
            -c "${language}_${bdata}_cuts" \
            -d $manifest_dir/$bdata \
            -s $sampling_rate \
            -o $cut_dir \
            -p \
            -v
    log "Succesfully stored $cut_dir/${language}_${bdata}_cuts.jsonl.gz"
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Stage 5: Dump transcripts from training and validation sets for LM"
    mkdir -p $lm_dir
    
    dev_args=""
    [ ! -z $kaldi_val_dir ] && dev_args="$kaldi_val_dir/text"
    #echo "$dev_args"

    cut -d' ' -sf2- $kaldi_train_dir/text $dev_args \
         > $lm_dir/transcript_words.txt
        
    tr ' ' '\n' < $lm_dir/transcript_words.txt \
        | sed -r "/^\s*$/d" \
        | sort -u > $lm_dir/words_full.txt
    
    # Separate regular and special words. The special words are enclosed in
    # <> or []. E.g. <filler>, [unintelligible/], etc.
    (echo "!SIL"; echo "<unk>"; (
        grep '<.*>\|\[.*\]' $lm_dir/words_full.txt | grep -v "<unk>\|<eps>")) \
        > $lm_dir/words_special.txt
    grep -v '<.*>\|\[.*\]' $lm_dir/words_full.txt | grep -v "!SIL" \
        > $lm_dir/words.txt
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    log "Stage 6: Prepare BPE based lang"
    mkdir -p $lang_dir

    # Prepare words.txt
    (echo "<eps>"; cat $lm_dir/words_special.txt $lm_dir/words.txt; \
        echo "#0"; echo "<s>"; echo "</s>") | awk '{print $0, NR-1}' \
        >$lang_dir/words.txt

    if [ ! -f $lang_dir/bpe.model ]; then
        $local/train_bpe_model.py \
        --lang-dir $lang_dir \
        --vocab-size $bpe_vocab_size \
        --transcript $lm_dir/transcript_words.txt \
        --special-words $lm_dir/words_special.txt
    fi

    if [ ! -f $lang_dir/L_disambig.pt ]; then
        # TODO: This script does not add entries for special words.
        # So they are treated as <unk> by default. Change this behaviour.
        #$local/prepare_lang_bpe.py --lang-dir $lang_dir \
        $local/prepare_lang_bpe.py  --lang-dir $lang_dir \
            --special-words $lm_dir/words_special.txt

        log "Validating $lang_dir/lexicon.txt"
        $local/validate_bpe_lexicon.py \
        --lexicon $lang_dir/lexicon.txt \
        --bpe-model $lang_dir/bpe.model
    fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    log "Stage 7: Prepare arpa file"
    $shared/make_kn_lm.py \
        -ngram-order 3 \
        -text $lm_dir/transcript_words.txt \
        -lm $lm_dir/G_3_gram.arpa
    # lmplz \
    #     --text $lang_dir/transcript_words.txt \
    #     -o 3 \
    #     --arpa $lm_dir/G_3_gram.arpa

fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    log "Stage 8: Prepare G"
    # We assume you have install kaldilm, if not, please install
    # it using: pip install kaldilm

    mkdir -p $lm_dir
    if [ ! -f $lm_dir/G_3_gram.fst.txt ]; then
        # It is used in building HLG
        python3 -m kaldilm \
        --read-symbol-table="$lang_dir/words.txt" \
        --disambig-symbol='#0' \
        --max-order=3 \
        $lm_dir/G_3_gram.arpa > $lm_dir/G_3_gram.fst.txt
    fi
fi

# Compile LG for RNN-T fast_beam_search decoding
if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    log "Stage 9: Compile LG"
    #$local/compile_lg.py --lang-dir $lang_dir
    $local/compile_lg.py --lang-dir $lang_dir --lm-dir $lm_dir
fi

# Extract features
if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    log "Stage 10: Extract features"
    
    if $on_the_fly_feats; then
        log "Skipping since on the fly features are enabled"
    else
        log "Processing musan"
        $scripts/compute_store_fbank_cuts.py \
            -i "$cut_dir/musan_cuts.jsonl.gz" \
            -o "$cut_dir/.temp_new_musan_cuts.jsonl.gz" \
            -f "$cut_dir/musan_features" \
            -n $feature_dim \
            -j $num_jobs

            mv $cut_dir/.temp_new_musan_cuts.jsonl.gz $cut_dir/musan_cuts.jsonl.gz
        log "Successfully updated musan cuts"

        for data in $kaldi_test_dirs $kaldi_val_dir $kaldi_train_dir; do
            bdata=$(basename $data)
            log "Processing $bdata"
            $scripts/compute_store_fbank_cuts.py \
                -i $cut_dir/${language}_${bdata}_cuts.jsonl.gz \
                -o $cut_dir/.temp_new_${language}_${bdata}_cuts.jsonl.gz \
                -f $cut_dir/${language}_${bdata}_features \
                -n $feature_dim \
                -j $num_jobs
            
            mv $cut_dir/.temp_new_${language}_${bdata}_cuts.jsonl.gz $cut_dir/${language}_${bdata}_cuts.jsonl.gz
            log "Succesfully updated $bdata cuts"
        done
    fi
fi

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
    log "Stage 11: Extracting dev set from train set, if specified "
# when Kaldi dev directory is empty, we will take out part of train set and prepare dev set
    if [ -z $kaldi_val_dir ] && [ $prepare_dev_from_train == "true" ]; then 
        log "Building dev set from train set (Splitting train set into train & dev set)"
        train_bdata=$(basename $kaldi_train_dir)
        full_train_cutset=$cut_dir/${language}_${train_bdata}_cuts.jsonl.gz
        train_cutset=$cut_dir/${language}_train_${train_bdata}_cuts.jsonl.gz
        dev_cutset=$cut_dir/${language}_dev_${train_bdata}_cuts.jsonl.gz

        full_tr_num_lines=`zcat ${full_train_cutset} | wc -l` 
        dev_file_lines=`echo "($full_tr_num_lines * $dev_from_train_percent)/100" | bc`
        let tr_file_lines=$full_tr_num_lines-$dev_file_lines

        log "Count of cuts before splitting : Train set = $tr_file_lines , Dev set = $dev_file_lines , & Full train set (before splitting) = $full_tr_num_lines"
        zcat ${full_train_cutset} | shuf > .shuflled_full_train_cutset
        
        split -l "$tr_file_lines" -a 1 -d .shuflled_full_train_cutset .xzl_temp

        gzip .xzl_temp0 .xzl_temp1
        mv .xzl_temp0.gz ${train_cutset}
        mv .xzl_temp1.gz ${dev_cutset}

         log "Count of cuts after splitting  : Train set = `zcat ${train_cutset} | wc -l` , Dev set = `zcat ${dev_cutset} | wc -l` "
        log "stored train at ${train_cutset} "
        log "stored dev at ${dev_cutset}"
        new_full_train=`dirname ${full_train_cutset}`/full_`basename ${full_train_cutset}`
        log "Moving ${full_train_cutset} --> ${new_full_train}"

        mv ${full_train_cutset}  ${new_full_train}

        rm .shuflled_full_train_cutset 
    fi
fi

