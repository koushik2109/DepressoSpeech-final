#!/bin/bash

################################################################################
# DepressoSpeech - Start All Services
# This script starts Frontend, Backend, and ML Model services
#
# Usage:
#   ./start-all.sh                    # Start with dependency check
#   ./start-all.sh --no-deps          # Start without installing dependencies
#   ./start-all.sh --kill-only        # Only kill existing processes
#
# Services will run on:
#   - Frontend: http://localhost:5173
#   - Backend: http://localhost:8000
#   - ML Model: http://localhost:8001
#   - Swagger:  http://localhost:8080
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/Depression-UI"
BACKEND_DIR="$SCRIPT_DIR/backend"
MODEL_DIR="$SCRIPT_DIR/Model"
SWAGGER_DIR="$SCRIPT_DIR/swagger"

FRONTEND_PORT=5173
BACKEND_PORT=8000
MODEL_PORT=8001
SWAGGER_PORT=8080

INSTALL_DEPS=true
KILL_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-deps)
            INSTALL_DEPS=false
            shift
            ;;
        --kill-only)
            KILL_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-deps] [--kill-only]"
            exit 1
            ;;
    esac
done

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}➜ $1${NC}"
}

# Function to check if a port is in use
is_port_in_use() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    elif fuser $port/tcp >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to get PIDs by port (tries lsof then fuser)
get_pids_by_port() {
    local port=$1
    local pids
    pids=$(lsof -i :$port -t 2>/dev/null | tr '\n' ' ')
    if [ -z "$pids" ]; then
        pids=$(fuser $port/tcp 2>/dev/null | tr -s ' ')
    fi
    echo "$pids"
}

# Function to kill process on port
kill_port() {
    local port=$1
    if is_port_in_use $port; then
        print_info "Port $port is in use. Killing process..."
        local pids=$(get_pids_by_port $port)
        if [ ! -z "$pids" ]; then
            kill -9 $pids 2>/dev/null || true
            sleep 1
            if is_port_in_use $port; then
                print_error "Failed to kill process on port $port"
                return 1
            else
                print_success "Killed process on port $port"
                return 0
            fi
        fi
    else
        print_success "Port $port is free"
        return 0
    fi
}

## Function to check NodeJS installation
check_nodejs() {
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed"
        echo "Please install Node.js from https://nodejs.org/"
        return 1
    fi
    print_success "Node.js found: $(node --version)"
    return 0
}

# Function to check Python installation
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        echo "Please install Python 3"
        return 1
    fi
    print_success "Python 3 found: $(python3 --version)"
    return 0
}

# Function to check and start PostgreSQL
check_postgres() {
    print_header "Checking PostgreSQL"

    # Ensure the service is running
    if command -v pg_isready &> /dev/null; then
        if pg_isready -h 127.0.0.1 -p 5432 -q 2>/dev/null; then
            print_success "PostgreSQL is running on 127.0.0.1:5432"
            return 0
        fi
    fi

    # Try to start via systemctl (Ubuntu / Debian)
    if command -v systemctl &> /dev/null && systemctl list-units --type=service | grep -q postgresql; then
        print_info "Starting PostgreSQL service..."
        sudo systemctl start postgresql 2>/dev/null || true
        sleep 2
        if pg_isready -h 127.0.0.1 -p 5432 -q 2>/dev/null; then
            print_success "PostgreSQL started successfully"
            return 0
        fi
    fi

    # Try pg_ctlcluster (Ubuntu multi-version setup)
    if command -v pg_ctlcluster &> /dev/null; then
        PG_VER=$(ls /etc/postgresql/ 2>/dev/null | sort -V | tail -1)
        if [ -n "$PG_VER" ]; then
            print_info "Trying pg_ctlcluster $PG_VER main start..."
            sudo pg_ctlcluster "$PG_VER" main start 2>/dev/null || true
            sleep 2
        fi
    fi

    if pg_isready -h 127.0.0.1 -p 5432 -q 2>/dev/null; then
        print_success "PostgreSQL is now running"
        return 0
    fi

    print_error "Could not start PostgreSQL — backend may fail to connect."
    print_error "Run: sudo systemctl start postgresql"
    return 1
}

# Function to install npm dependencies
install_npm_deps() {
    print_header "Installing Frontend Dependencies (npm)"
    if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
        cd "$FRONTEND_DIR"
        print_info "Running: npm install"
        npm install --legacy-peer-deps
        print_success "Frontend dependencies installed"
        cd "$SCRIPT_DIR"
    else
        print_success "Frontend dependencies already installed"
    fi
}

# Function to install Python dependencies for Backend (uses backend venv)
install_backend_deps() {
    print_header "Installing Backend Dependencies (pip into .venv)"

    local PIP_CMD="pip3"
    if [ -f "$BACKEND_DIR/.venv/bin/pip" ]; then
        PIP_CMD="$BACKEND_DIR/.venv/bin/pip"
    else
        # Create the venv if it doesn't exist
        print_info "Creating backend virtual environment..."
        python3 -m venv "$BACKEND_DIR/.venv"
        PIP_CMD="$BACKEND_DIR/.venv/bin/pip"
    fi

    if $PIP_CMD show fastapi > /dev/null 2>&1; then
        print_success "Backend dependencies already installed"
    else
        cd "$BACKEND_DIR"
        print_info "Running: pip install -r requirements.txt (in backend .venv)"
        $PIP_CMD install -r requirements.txt
        print_success "Backend dependencies installed"
        cd "$SCRIPT_DIR"
    fi
}

# Function to install Python dependencies for Model (uses Model venv)
install_model_deps() {
    print_header "Installing ML Model Dependencies (pip)"

    local PIP_CMD="pip"
    if [ -f "$MODEL_DIR/.venv/bin/pip" ]; then
        PIP_CMD="$MODEL_DIR/.venv/bin/pip"
    fi

    if $PIP_CMD show slowapi > /dev/null 2>&1; then
        print_success "ML Model dependencies already installed"
    else
        cd "$MODEL_DIR"
        print_info "Running: pip install -r requirements.txt (in Model venv)"
        $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        $PIP_CMD install -r requirements.txt
        print_success "ML Model dependencies installed"
        cd "$SCRIPT_DIR"
    fi
}

# Function to start frontend
start_frontend() {
    print_header "Starting Frontend (Vite Dev Server)"
    cd "$FRONTEND_DIR"
    print_info "Frontend will run on: http://localhost:$FRONTEND_PORT"
    npm run dev > /tmp/frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd "$SCRIPT_DIR"
    sleep 2
}

# Function to start backend
start_backend() {
    print_header "Starting Backend (FastAPI - MindScope)"
    print_info "Backend will run on: http://localhost:$BACKEND_PORT"

    # Prefer the local venv created inside the backend dir
    local UVICORN_CMD
    if [ -f "$BACKEND_DIR/.venv/bin/uvicorn" ]; then
        print_info "Using backend venv: $BACKEND_DIR/.venv"
        UVICORN_CMD="$BACKEND_DIR/.venv/bin/uvicorn"
    else
        print_info "No backend venv found, falling back to system python3"
        UVICORN_CMD="python3 -m uvicorn"
    fi

    (cd "$BACKEND_DIR" && $UVICORN_CMD main:app --host 0.0.0.0 --port $BACKEND_PORT --reload > /tmp/backend.log 2>&1) &
    BACKEND_PID=$!
    sleep 3
}

# Function to start model server (activates Model venv)
start_model() {
    print_header "Starting ML Model Server (DepressoSpeech API)"
    print_info "ML Model Server will run on: http://localhost:$MODEL_PORT"
    print_info "Note: first startup may take longer while sentence-transformers models are downloaded."

    if [ -f "$MODEL_DIR/.venv/bin/activate" ]; then
        print_info "Activating Model virtual environment..."
        (cd "$MODEL_DIR" && source .venv/bin/activate && python scripts/serve.py --port $MODEL_PORT > /tmp/model_serve.log 2>&1) &
    else
        print_info "No Model venv found, using system Python..."
        (cd "$MODEL_DIR" && python3 scripts/serve.py --port $MODEL_PORT > /tmp/model_serve.log 2>&1) &
    fi
    MODEL_PID=$!
    sleep 10
}

# Function to start swagger docs server
start_swagger() {
    print_header "Starting Swagger Documentation Server"
    print_info "Swagger UI will run on: http://localhost:$SWAGGER_PORT"
    (cd "$SWAGGER_DIR" && python3 "$SWAGGER_DIR/serve.py" --port $SWAGGER_PORT > /tmp/swagger.log 2>&1) &
    SWAGGER_PID=$!
    sleep 1
}

# Function to check service status (with retries for slow starters)
check_service_status() {
    local service=$1
    local port=$2
    local log_file=$3
    local retries=12

    for i in $(seq 1 $retries); do
        if is_port_in_use $port; then
            print_success "$service is running on port $port"
            return 0
        fi
        [ $i -lt $retries ] && sleep 2
    done

    print_error "$service failed to start on port $port"
    if [ -f "$log_file" ]; then
        echo -e "\n${YELLOW}Last 10 lines from log:${NC}"
        tail -10 "$log_file"
    fi
    return 1
}

# Function to show final status
show_final_status() {
    print_header "Service Status"

    echo -e "${BLUE}Frontend (Vite)${NC}"
    check_service_status "Frontend" "$FRONTEND_PORT" "/tmp/frontend.log" || true

    echo ""
    echo -e "${BLUE}Backend (FastAPI - MindScope)${NC}"
    check_service_status "Backend" "$BACKEND_PORT" "/tmp/backend.log" || true

    echo ""
    echo -e "${BLUE}ML Model Server (DepressoSpeech API)${NC}"
    check_service_status "ML Model" "$MODEL_PORT" "/tmp/model_serve.log" || true

    echo ""
    echo -e "${BLUE}Swagger Documentation Server${NC}"
    check_service_status "Swagger" "$SWAGGER_PORT" "/tmp/swagger.log" || true

    echo ""
    print_header "Access Points"
    echo -e "${GREEN}Frontend:        ${NC}http://localhost:$FRONTEND_PORT"
    echo -e "${GREEN}Backend API:     ${NC}http://localhost:$BACKEND_PORT/api/v1"
    echo -e "${GREEN}Backend Docs:    ${NC}http://localhost:$BACKEND_PORT/docs"
    echo -e "${GREEN}Model API:       ${NC}http://localhost:$MODEL_PORT"
    echo -e "${GREEN}Model Docs:      ${NC}http://localhost:$MODEL_PORT/docs"
    echo -e "${GREEN}Swagger UI:      ${NC}http://localhost:$SWAGGER_PORT"

    echo ""
    print_header "Log Files"
    echo -e "Frontend:  /tmp/frontend.log"
    echo -e "Backend:   /tmp/backend.log"
    echo -e "Model:     /tmp/model_serve.log"
    echo -e "Swagger:   /tmp/swagger.log"

    echo ""
    print_header "Database (PostgreSQL)"
    echo -e "${GREEN}Host:         ${NC}127.0.0.1:5432"
    echo -e "${GREEN}Database:     ${NC}mindscope"
    echo -e "${GREEN}User:         ${NC}mindscope"
    echo -e "${GREEN}Type:         ${NC}PostgreSQL 16 (asyncpg driver)"
    echo -e "${GREEN}File storage: ${NC}BYTEA (audio/video stored in media_file_data table)"
    echo -e "${GREEN}Tables:       ${NC}users, doctors, assessments, media_files, media_file_data, etc."
    echo ""
    echo -e "${YELLOW}To inspect the database:${NC}"
    echo -e "  psql -h 127.0.0.1 -U mindscope -d mindscope"
    echo -e "  SELECT table_name FROM information_schema.tables WHERE table_schema='public';"

    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}\n"
}

################################################################################
# Main Script
################################################################################

print_header "DepressoSpeech - Start All Services"

# Check if running from correct directory
if [ ! -d "$FRONTEND_DIR" ] || [ ! -d "$BACKEND_DIR" ] || [ ! -d "$MODEL_DIR" ]; then
    print_error "Not running from DepressoSpeech root directory"
    print_error "Frontend dir: $FRONTEND_DIR"
    print_error "Backend dir: $BACKEND_DIR"
    print_error "Model dir: $MODEL_DIR"
    exit 1
fi

print_success "Running from: $SCRIPT_DIR"

# Kill existing processes on ports
print_header "Checking and Clearing Ports"
kill_port $FRONTEND_PORT || exit 1
kill_port $BACKEND_PORT || exit 1
kill_port $MODEL_PORT || exit 1
kill_port $SWAGGER_PORT || exit 1

# Exit if kill-only flag was set
if [ "$KILL_ONLY" = true ]; then
    print_success "All ports cleared. Exiting."
    exit 0
fi

# Disable exit-on-error for service startup (background processes)
set +e

# Check prerequisites
print_header "Checking Prerequisites"
check_nodejs || exit 1
check_python || exit 1
check_postgres || true   # Non-fatal: warn but proceed

# Install dependencies if needed
if [ "$INSTALL_DEPS" = true ]; then
    install_npm_deps
    install_backend_deps
    install_model_deps
else
    print_info "Skipping dependency installation (--no-deps flag set)"
fi

# Start all services
print_header "Starting Services"
start_frontend
start_backend
start_model
start_swagger

# Show status
show_final_status

# Keep script running to handle signals
trap 'print_error "Shutting down..."; kill $FRONTEND_PID $BACKEND_PID $MODEL_PID $SWAGGER_PID 2>/dev/null; exit 0' INT TERM

# Wait for all background processes
wait
