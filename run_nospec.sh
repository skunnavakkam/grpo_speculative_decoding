
# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    pip install uv
fi

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo ".venv not found, creating virtual environment..."
    uv venv
fi


# Install dependencies from requirements.txt
uv pip install -r requirements.txt

# Activate virtual environment
source .venv/bin/activate

python -m sglang.launch_server \
    --model_path Qwen/Qwen3-4B \ 
    --base-gpu-id 0 \

python sglang_impl_nospec.py