# Check if Git is installed
$gitPath = "C:\Program Files\Git\bin\git.exe"
if (-not (Test-Path $gitPath)) {
    Write-Host "Git not found. Please check your installation."
    Read-Host "Press Enter to exit..."
    exit 1
}

# Add Git to PATH for this session
$env:PATH += ";C:\Program Files\Git\bin"

# Show Git version
Write-Host "Git installed successfully! Version:"
& $gitPath --version
Write-Host

# Configure Git user name if not set
$username = & $gitPath config --global user.name 2>$null
if (-not $username) {
    Write-Host "Configuring Git user name..."
    & $gitPath config --global user.name "Huoguanhua"
    $username = "Huoguanhua"
}

# Configure Git email if not set
$email = & $gitPath config --global user.email 2>$null
if (-not $email) {
    Write-Host "Configuring Git email..."
    & $gitPath config --global user.email "hgh@example.com"
    $email = "hgh@example.com"
}

# Show configuration
Write-Host "Git configuration:"
Write-Host "User name: $username"
Write-Host "Email: $email"
Write-Host

# Start Git Bash
Write-Host "Starting Git Bash..."
Start-Process "C:\Program Files\Git\git-bash.exe"

Write-Host "Git Bash has been started!"
Write-Host "You can now use Git and Git Bash for version control."
Write-Host
Write-Host "Common Git commands:"
Write-Host "  git status     - Check current repository status"
Write-Host "  git clone      - Clone a remote repository"
Write-Host "  git add        - Add files to staging area"
Write-Host "  git commit     - Commit changes"
Write-Host "  git push       - Push to remote repository"
Write-Host "  git pull       - Pull from remote repository"
Write-Host "  git branch     - View branches"
Write-Host "  git checkout   - Switch branches"
Write-Host
Read-Host "Press Enter to exit..."