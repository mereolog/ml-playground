# Guide to Installing Docker

This guide provides easy-to-follow steps for installing Docker on Windows, macOS, and Ubuntu (or similar Linux distributions).

## What is Docker?

Think of Docker as a tool that lets you package applications and their dependencies into neat little boxes called "containers". These containers can run almost anywhere – on your laptop, on a server, in the cloud – ensuring your application works the same way regardless of the environment.

## 1. Installing Docker on Windows

The easiest way to get Docker on Windows is by using **Docker Desktop**.

**System Requirements:**

- Windows 10 or Windows 11 64-bit (Pro, Enterprise, Education) - it is important that you are using one of those Windows versions, other versions doesn't allow using Hyper-V
- Windows 10/11 Home requires **WSL 2** (Windows Subsystem for Linux version 2). Docker Desktop will often help you install this if needed.
- Hardware virtualization must be enabled in your computer's BIOS/UEFI (it usually is by default on modern machines).

**Steps:**

1. **Download Docker Desktop:**

   - Go to the official Docker website: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
   - Click the download button for Windows.

2. **Run the Installer:**

   - Find the downloaded `.exe` file (usually in your `Downloads` folder) and double-click it.
   - You might be asked for administrator permission. Click "Yes".

3. **Follow Installation Prompts:**

   - The installer will guide you. Ensure the "Install required Windows components for WSL 2" option is checked if you're on Windows Home or if it's recommended.
   - Click "Ok" or "Next" through the configuration steps.

4. **Restart Your Computer:**

   - Once the installation is complete, you'll likely need to restart Windows.

5. **Launch Docker Desktop:**

   - After restarting, find "Docker Desktop" in your Start Menu and run it.
   - It might take a minute or two to start the Docker engine for the first time.
   - Accept the terms of service.
   - You might be prompted to log in with a Docker Hub account (optional, but useful for accessing more features later).

6. **Done!** Docker should now be running (you'll usually see a whale icon in your system tray).

## 2. Installing Docker on macOS

Similar to Windows, the recommended way on macOS is **Docker Desktop**.

**System Requirements:**

- macOS version that meets Docker's current requirements (check the Docker Desktop download page for specifics, usually the latest few major versions).
- A Mac with either an Intel chip or an Apple Silicon chip (M1, M2, etc.).

**Steps:**

1. **Download Docker Desktop:**

   - Go to the official Docker website: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
   - Click the download button for Mac. Make sure to choose the correct version for your chip (**Intel** or **Apple Silicon**).

2. **Install the Application:**

   - Find the downloaded `.dmg` file (usually in your `Downloads` folder) and double-click it.
   - A window will appear showing the Docker icon and your Applications folder.
   - Drag the Docker icon into the Applications folder.

3. **Launch Docker Desktop:**

   - Open your `Applications` folder and double-click "Docker".
   - You might be asked if you're sure you want to open an application downloaded from the internet. Click "Open".
   - You may also need to grant Docker privileged access (permission to manage networking, etc.). Enter your Mac password if prompted.

4. **Accept Terms and Start:**

   - Accept the terms of service.
   - Docker Desktop will start. You'll see the whale icon in your top menu bar.
   - You might be prompted to log in with a Docker Hub account (optional).

5. **Done!** Docker is ready to use.

## 3. Installing Docker on Ubuntu (or similar Linux)

For Ubuntu and related distributions (like Debian, Linux Mint), you'll install **Docker Engine** using the command line. It's best to use Docker's official repository for the latest updates.

**Steps:**

1. **Update Package List:**
   Open your terminal and run:

   ```bash
   sudo apt update
   ```

2. **Install Prerequisite Packages:**
   These allow `apt` to use repositories over HTTPS:

   ```bash
   sudo apt install -y apt-transport-https ca-certificates curl software-properties-common gnupg lsb-release
   ```

   _(Note: `-y` automatically confirms installation)_

3. **Add Docker's Official GPG Key:**
   This verifies the authenticity of the Docker packages:

   ```bash
   sudo mkdir -p /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   ```

4. **Set Up the Docker Repository:**
   This tells `apt` where to download Docker from:

   ```bash
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

5. **Update Package List Again (with Docker repo):**

   ```bash
   sudo apt update
   ```

6. **Install Docker Engine:**
   Install the latest version of Docker Engine, CLI, containerd, and useful plugins:

   ```bash
   sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

7. **Verify Installation:**
   Check if Docker is running by trying to run the `hello-world` container:

   ```bash
   sudo docker run hello-world
   ```

   You should see a message indicating that your installation appears to be working correctly.

8. **(Optional but Recommended) Run Docker without `sudo`:**
   By default, you need `sudo` to run Docker commands. To allow your user to run Docker commands without `sudo`:

   - Add your user to the `docker` group (this group was created during installation):

     ```bash
     sudo usermod -aG docker $USER
     ```

   - **Important:** Log out and log back in for this change to take effect. Alternatively, you can run `newgrp docker` in your current terminal session to apply the group membership temporarily.
   - After logging back in, verify you can run Docker without `sudo`:

     ```bash
     docker run hello-world
     ```

9. **Done!** Docker Engine is installed and ready on your Linux machine.

## Verifying Your Installation (All Platforms)

After following the steps for your OS, open a terminal or command prompt and run these commands:

1. **Check Docker version:**

   ```bash
   docker --version
   ```

   This should output the installed Docker version.

2. **Check Docker Compose version** (if installed, included with Docker Desktop and via `docker-compose-plugin` on Linux):

   ```bash
   docker compose version
   ```

3. **Run the `hello-world` image:**

   ```bash
   docker run hello-world
   ```

   This command downloads a tiny test image and runs it in a container. If it works, you'll see a "Hello from Docker!" message.

You now have Docker installed! You can start exploring Docker Hub for images or learn basic Docker commands like `docker ps` or `docker compose`.
