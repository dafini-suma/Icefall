#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=-1
stop_stage=100
perturb_speed=true


# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/aidatatang_200zh
#      You can find "corpus" and "transcript" inside it.
#      You can download it at https://openslr.org/62/
#      If you download the data by yourself, DON'T FORGET to extract the *.tar.gz files under corpus.

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  if [ ! -f $dl_dir/aidatatang_200zh/transcript/aidatatang_200_zh_transcript.txt ]; then
    lhotse download aidatatang-200zh $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare aidatatang_200zh manifest"
  # We assume that you have downloaded the aidatatang_200zh corpus
  # to $dl_dir/aidatatang_200zh
  if [ ! -f data/manifests/aidatatang_200zh/.manifests.done ]; then
    mkdir -p data/manifests/aidatatang_200zh
    lhotse prepare aidatatang-200zh $dl_dir data/manifests/aidatatang_200zh
    touch data/manifests/aidatatang_200zh/.manifests.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  if [ ! -f data/manifests/.manifests.done ]; then
    log "It may take 6 minutes"
    mkdir -p data/manifests/
    lhotse prepare musan $dl_dir/musan data/manifests/
    touch data/manifests/.manifests.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute fbank for musan"
  if [ ! -f data/fbank/.msuan.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_musan.py
    touch data/fbank/.msuan.done
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for aidatatang_200zh"
  if [ ! -f data/fbank/.aidatatang_200zh.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_aidatatang_200zh.py --perturb-speed ${perturb_speed}
    touch data/fbank/.aidatatang_200zh.done
  fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Prepare char based lang"
  lang_char_dir=data/lang_char
  mkdir -p $lang_char_dir
  # Prepare text.
  # Note: in Linux, you can install jq with the following command:
  # 1. wget -O jq https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64
  # 2. chmod +x ./jq
  # 3. cp jq /usr/bin
  if [ ! -f $lang_char_dir/text ]; then
    gunzip -c data/manifests/aidatatang_200zh/aidatatang_supervisions_train.jsonl.gz \
      |jq '.text' |sed -e 's/["text:\t ]*//g' | sed 's/"//g' \
      | ./local/text2token.py -t "char" > $lang_char_dir/text
  fi
  # Prepare words.txt
  if [ ! -f $lang_char_dir/text_words ]; then
    gunzip -c data/manifests/aidatatang_200zh/aidatatang_supervisions_train.jsonl.gz \
      | jq '.text' | sed -e 's/["text:\t]*//g' | sed 's/"//g' \
      | ./local/text2token.py -t "char" > $lang_char_dir/text_words
  fi

  cat $lang_char_dir/text_words | sed 's/ /\n/g' | sort -u | sed '/^$/d' \
    | uniq > $lang_char_dir/words_no_ids.txt

  if [ ! -f $lang_char_dir/words.txt ]; then
    ./local/prepare_words.py \
      --input-file $lang_char_dir/words_no_ids.txt \
      --output-file $lang_char_dir/words.txt
  fi

  if [ ! -f $lang_char_dir/L_disambig.pt ]; then
    ./local/prepare_char.py
  fi
fi
