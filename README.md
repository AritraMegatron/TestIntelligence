# Test Intelligence

This application uses artificial intelligence to assist with test planning and execution.

## Getting Started with Local AI (Ollama)

We have updated the application to run the AI model completely locally on your machine instead of using the cloud-based OpenAI API. This ensures maximum privacy and removes any API costs. You no longer need an OpenAI API key, and the `openai` Python package is no longer required.

Follow these steps to set up your local environment:

### Step 1: Install Ollama
Ollama is the engine that runs the AI models locally on your machine.
1. Go to [ollama.com/download](https://ollama.com/download)
2. Download the installer for your operating system (Windows, Mac, or Linux) and run it.
3. Once installed, Ollama will run in the background (you might see a new icon in your system tray/menu bar).

### Step 2: Download the AI Model
By default, the application is configured to use **Llama 3**. You need to download this model to your machine once.
1. Open your terminal or command prompt.
2. Run the following command:
   ```bash
   ollama run llama3
   ```
3. *Note: This will take a few minutes as it downloads the model weights (a few gigabytes). Once it finishes and you see a `>>>` chat prompt, you can just type `/bye` to exit. The model is now saved on your machine.*

### Step 3: Set up your Python Environment
Make sure your Python environment is up to date:
1. Ensure your virtual environment is activated (e.g. `venv\Scripts\activate` on Windows).
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

### Step 4: Configure the Environment Variables
You no longer need a real `.env` file for the API key, but the application expects to know which local model to use.
1. Create a file named `.env` in the root folder of the project.
2. Add the following line to it:
   ```env
   OPENAI_MODEL=llama3
   ```
   *(Note: The variable is still named `OPENAI_MODEL` for legacy compatibility in the code, but it is now pointing to your local Ollama model).*

### Step 5: Run the Application
Start the application exactly as you normally would. As long as the Ollama app is running in the background on your computer, the application will automatically connect to it via `http://localhost:11434` and generate responses locally.
