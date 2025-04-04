# HealthGPT Medical Imaging Suite

A modern web application for enhancing medical images and generating diagnostic reports using AI.

## Features

- **Super-Resolution Enhancement**: Improve the resolution and clarity of low-quality medical images
- **Modality Conversion**: Convert between different imaging modalities (CT to MRI or MRI to CT)
- **Diagnostic Analysis**: Generate detailed diagnostic reports from medical images
- **Interactive UI**: Side-by-side comparison of original and processed images
- **Dark Mode Support**: Toggle between light and dark themes

## Tech Stack

- **Frontend**: Next.js, React, Ant Design, Tailwind CSS
- **Backend**: Python FastAPI server with HealthGPT model integration
- **Deployment**: Vercel (frontend), Hugging Face Inference Endpoints (model)

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+ (for backend)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/healthgpt-medical-imaging.git
   cd healthgpt-medical-imaging
   ```

2. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd ../backend
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```bash
   python api_server.py
   ```

## Usage

1. Upload a medical image (JPG, PNG, or DICOM format)
2. Select a task type:
   - Super-Resolution Enhancement
   - CT to MRI Conversion
   - MRI to CT Conversion
   - Diagnostic Analysis
3. Click "Process Scan" to start the processing
4. View the results in the appropriate tab:
   - Image Comparison tab for enhanced/converted images
   - Diagnostic Report tab for analysis results

## Project Structure

```
frontend/
├── public/              # Static assets
├── src/
│   ├── app/             # Next.js app router
│   ├── components/      # React components
│   │   ├── Dashboard.js       # Main dashboard layout
│   │   ├── UploadForm.js      # Image upload and processing
│   │   ├── ImageComparison.js # Side-by-side image comparison
│   │   ├── DiagnosticReport.js # Diagnostic report display
│   │   └── Settings.js        # Application settings
│   └── context/         # React context providers
└── package.json         # Dependencies and scripts
```

## Deployment

### Frontend Deployment (Vercel)

1. Push your code to a GitHub repository
2. Connect the repository to Vercel
3. Configure environment variables if needed
4. Deploy

### Model Deployment (Hugging Face)

1. Create a Hugging Face account
2. Upload your model to Hugging Face
3. Deploy as an Inference Endpoint
4. Update the API endpoint in the frontend settings

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HealthGPT team for the underlying AI models
- Ant Design for the UI components
- Next.js team for the React framework
