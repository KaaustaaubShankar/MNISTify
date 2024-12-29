# MNISTify - Digit Recognition with LIME Explanations

MNISTify is a web app for recognizing hand-drawn digits using deep learning and providing LIME-based explanations.

## Features
- Real-time digit recognition
- LIME-based AI explanations
- Dockerized backend (Flask) and frontend (Svelte)

## Tech Stack
- **Frontend**: Svelte, TailwindCSS
- **Backend**: Flask, PyTorch
- **AI**: MNIST CNN, LIME
- **Deployment**: Docker, Docker Compose

## Prerequisites
- Docker & Docker Compose
- Git

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KaaustaaubShankar/MNISTify.git
   cd mnistify
   ```

2. **Build containers**:
   ```bash
   docker-compose build
   ```

3. **Run the application**:
   ```bash
   docker-compose up
   ```

4. **Access the app**:
   - Flask backend: [http://localhost:8000](http://localhost:8000)
   - Svelte frontend: [http://localhost:5001](http://localhost:5001)
     
5. **Stop the app**:
   ```bash
   docker-compose down
   ```

## Docker Setup

### Backend (Flask)
The Flask API handles digit recognition and LIME explanations.

### Frontend (Svelte)
Svelte app provides an interactive canvas for drawing digits and is located at [http://localhost:5001](http://localhost:5001).

## License
MIT License.

