# HealthGPT Medical Imaging Suite

This project is an enhanced medical imaging and diagnostic assistant that uses HealthGPT models to improve low-resolution medical images, convert between imaging modalities, and generate diagnostic reports.

## Features

- **Image Enhancement**: Improve the resolution and clarity of low-quality medical images
- **Modality Conversion**: Convert between CT and MRI scans
- **Diagnostic Report Generation**: Analyze medical images and provide detailed diagnostic reports

## Prerequisites

- Python 3.10+
- Node.js 16+
- CUDA-compatible GPU (recommended for faster processing)

## Installation

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/HealthGPT-main.git
   cd HealthGPT-main
   ```

2. Create and activate a Python virtual environment:
   ```bash
   conda create -n HealthGPT python=3.10
   conda activate HealthGPT
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the required model weights:
   ```bash
   cd weights
   python download_weights.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install the required Node.js packages:
   ```bash
   npm install
   ```

## Usage

### Running the Backend

1. Start the backend server:
   ```bash
   python app.py
   ```

### Running the Frontend

1. In a separate terminal, navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Start the Next.js development server:
   ```bash
   npm run dev
   ```

3. Open your browser and navigate to `http://localhost:3000`

### Using the Application

1. Select a task type (Image Enhancement, CT to MRI Conversion, MRI to CT Conversion, or Generate Diagnosis)
2. Upload a medical image
3. Click "Process Image" to start the processing
4. View the results in the output section

## API Endpoints

The application provides the following API endpoints:

- `POST /api/process-scan`: Process a medical image based on the selected task type
  - Request body:
    - `scan`: The medical image file
    - `taskType`: The type of task to perform (`enhance`, `ct2mri`, `mri2ct`, or `diagnosis`)
  - Response:
    - `result`: The processed image (as base64) or diagnostic text

## Models Used

This project uses the following HealthGPT models:

- **HealthGPT-XL32**: For image enhancement, modality conversion, and diagnostic report generation
- **HealthGPT-M3**: For image generation tasks

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [HealthGPT](https://huggingface.co/lintw/HealthGPT-XL32) for providing the pre-trained models
- [Next.js](https://nextjs.org/) for the frontend framework
- [Tailwind CSS](https://tailwindcss.com/) for styling
