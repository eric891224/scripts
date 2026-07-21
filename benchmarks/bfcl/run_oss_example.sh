# make sure to follow the instructions in Notes/benchmarks/bfcl/ 
# to set up the custom model registry and the .env file before running this script

# MODEL_REGISTRY_NAME=gpt-oss-120b-fc
MODEL_REGISTRY_NAME=gpt-oss-120b-prompt

# TEST_CATEGORY=simple_python
TEST_CATEGORY=parallel

### Generation

# bfcl generate \
#   --model $MODEL_REGISTRY_NAME \
#   --test-category $TEST_CATEGORY \
#   --skip-server-setup \
#   --num-threads 1

bfcl generate \
  --model $MODEL_REGISTRY_NAME \
  --test-category $TEST_CATEGORY \
  --skip-server-setup \
  --num-threads 8

### Evaluation

bfcl evaluate \
  --model $MODEL_REGISTRY_NAME \
  --test-category $TEST_CATEGORY