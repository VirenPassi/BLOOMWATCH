# Running BloomWatch Locally

This guide will help you set up and run your BloomWatch FastAPI application locally on your Windows machine.

## Quick Start

### Option 1: PowerShell Script (Recommended)

1. **Open PowerShell** in the BloomWatch project directory (`D:\NASA(0)\BloomWatch`)

2. **Run the setup script:**
 ```powershell
 .\run_app.ps1
 ```

3. **If you get execution policy errors, run:**
 ```powershell
 Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
 .\run_app.ps1
 ```

### Option 2: Batch File (CMD)

1. **Open Command Prompt** in the BloomWatch project directory

2. **Run the batch file:**
 ```cmd
 run_app.bat
 ```

### Option 3: Manual Setup

If the automated scripts don't work, you can set up manually:

```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## What the Script Does

The `run_app.ps1` script automatically:

1. **Checks Python installation** (requires Python 3.9+)
2. **Creates virtual environment** (`.venv` folder)
3. **Activates virtual environment**
4. **Installs all dependencies** from `requirements.txt`
5. **Launches FastAPI server** with uvicorn
6. **Handles PowerShell execution policy** issues
7. **Provides clear output** and URLs

## Accessing Your App

Once the script runs successfully, you can access your BloomWatch app at:

- **Main App**: http://127.0.0.1:8000
- **API Documentation**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health
- **ReDoc Documentation**: http://127.0.0.1:8000/redoc

## Script Options

The PowerShell script supports several options:

```powershell
# Run with default settings
.\run_app.ps1

# Run on a different port
.\run_app.ps1 -Port 8080

# Use a different Python version
.\run_app.ps1 -PythonVersion 3.11

# Skip dependency installation (faster for testing)
.\run_app.ps1 -SkipDependencies

# Show help
.\run_app.ps1 -Help
```

## Troubleshooting

### Common Issues

1. **"Execution policy is set to Restricted"**
 ```powershell
 Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
 ```

2. **"Python not found"**
 - Install Python 3.9+ from https://python.org
 - Make sure Python is added to PATH during installation

3. **"pip not found"**
 - Reinstall Python with pip included
 - Or install pip manually: `python -m ensurepip --upgrade`

4. **"Failed to install dependencies"**
 - Check your internet connection
 - Try running: `pip install --upgrade pip`
 - Then run the script again

5. **"uvicorn not found"**
 - Install manually: `pip install uvicorn[standard]`

6. **Port already in use**
 - Use a different port: `.\run_app.ps1 -Port 8080`
 - Or stop the process using port 8000

### Getting Help

If you encounter issues:

1. **Check the terminal output** for specific error messages
2. **Try the manual setup** steps above
3. **Check Python version**: `python --version`
4. **Check pip version**: `pip --version`
5. **Verify you're in the right directory**: Should contain `requirements.txt` and `app\main.py`

## Development Tips

- **Auto-reload**: The server automatically reloads when you make changes to your code
- **Stop the server**: Press `Ctrl+C` in the terminal
- **View logs**: All server logs appear in the terminal
- **API testing**: Use the interactive docs at `/docs` to test your API endpoints

## Next Steps

Once your app is running:

1. **Test the API** using the interactive documentation at http://127.0.0.1:8000/docs
2. **Check health** at http://127.0.0.1:8000/health
3. **Start developing** - make changes to your code and see them reflected immediately
4. **Add your data** - use the MODIS data fetching tools to add real data

Happy coding! 
