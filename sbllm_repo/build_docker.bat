@echo off
REM RISC-V 优化框架 - Docker 镜像构建脚本
REM
REM 使用方法:
REM   build_docker.bat         - 构建镜像
REM   build_docker.bat --force - 强制重新构建

echo ========================================
echo RISC-V 优化框架 Docker 镜像构建
echo ========================================

cd /d "%~dp0"

REM 检查 Docker 是否可用
docker info >nul 2>&1
if errorlevel 1 (
    echo 错误: Docker 未运行或未安装
    echo 请确保 Docker Desktop 已启动
    exit /b 1
)

REM 检查是否强制重建
set BUILD_ARGS=
if "%1"=="--force" (
    echo 强制重新构建镜像...
    set BUILD_ARGS=--no-cache
)

REM 构建镜像
echo.
echo 正在构建 Docker 镜像 (riscv-opt-env:latest)...
echo 这可能需要 5-15 分钟，取决于网络速度...
echo.

docker build %BUILD_ARGS% -t riscv-opt-env:latest .

if errorlevel 1 (
    echo.
    echo ========================================
    echo 构建失败！
    echo ========================================
    echo 请检查网络连接或 Dockerfile 配置
    exit /b 1
)

echo.
echo ========================================
echo 构建成功！
echo ========================================
echo.
echo 镜像信息:
docker images riscv-opt-env:latest

echo.
echo 启动开发环境:
echo   docker run -it -v %cd%:/app --name project1-dev riscv-opt-env:latest
echo.
echo 或使用 start_dev.bat 脚本
