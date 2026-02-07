@echo off
set CONTAINER_NAME=project1-dev
set IMAGE_NAME=riscv-opt-env

rem Check if container exists
docker ps -a --format "{{.Names}}" | findstr /C:"%CONTAINER_NAME%" >nul
if %errorlevel%==0 (
    echo Container %CONTAINER_NAME% already exists.
    
    rem Check if it is running
    docker ps --format "{{.Names}}" | findstr /C:"%CONTAINER_NAME%" >nul
    if %errorlevel%==0 (
        echo Container is running. Entering...
        docker exec -it %CONTAINER_NAME% bash
    ) else (
        echo Container is stopped. Starting...
        docker start -ai %CONTAINER_NAME%
    )
) else (
    echo Creating new persistent container %CONTAINER_NAME%...
    docker run -it --name %CONTAINER_NAME% ^
        -v "%cd%":/app ^
        %IMAGE_NAME%
)
