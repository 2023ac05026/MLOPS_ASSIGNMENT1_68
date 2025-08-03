@echo off
ECHO Stopping and removing all services (MLflow server and prediction app)...

REM Stop and remove containers, networks defined in docker-compose.yml
docker-compose down

ECHO.
ECHO Services have been stopped.
ECHO.
PAUSE