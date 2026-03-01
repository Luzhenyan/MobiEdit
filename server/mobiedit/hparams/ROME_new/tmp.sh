i=0
# for f in qwen2.5-3b-*-zo\ copy.yaml; do
#   mv "$f" "qwen2.5-3b-${i}-zo-prefix.yaml"
#   i=$((i + 1))
# done

for f in qwen2.5-3b-*-zo-prefix.yaml; do
  sed -i \
    -e 's/^quantize: false/quantize: false/' \
    -e 's/^use_random_prefix: false/use_random_prefix: true/' \
    "$f"
done


