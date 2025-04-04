import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// Helper function to convert base64 to file
async function saveBase64ToFile(base64Data, filePath) {
  const base64Image = base64Data.split(';base64,').pop();
  await fs.promises.writeFile(filePath, base64Image, { encoding: 'base64' });
  return filePath;
}

// Helper function to convert file to base64
async function fileToBase64(filePath) {
  const fileBuffer = await fs.promises.readFile(filePath);
  return `data:image/png;base64,${fileBuffer.toString('base64')}`;
}

export async function POST(request) {
  try {
    const formData = await request.formData();
    const file = formData.get('scan');
    const taskType = formData.get('taskType');
    
    if (!file) {
      return NextResponse.json({ error: 'No file uploaded' }, { status: 400 });
    }

    // Create a temporary directory for processing
    const tempDir = path.join(process.cwd(), 'tmp');
    await fs.promises.mkdir(tempDir, { recursive: true });
    
    // Save the uploaded file
    const fileBuffer = await file.arrayBuffer();
    const filePath = path.join(tempDir, `input_${Date.now()}.png`);
    await fs.promises.writeFile(filePath, Buffer.from(fileBuffer));
    
    // Output path for the processed image
    const outputPath = path.join(tempDir, `output_${Date.now()}.png`);
    
    // Process the image based on task type
    let result;
    
    switch (taskType) {
      case 'enhance':
        // Image enhancement using HealthGPT-XL32
        await execAsync(`python app.py --task_type enhancement --input_path ${filePath} --output_path ${outputPath}`);
        result = await fileToBase64(outputPath);
        break;
        
      case 'ct2mri':
        // CT to MRI conversion using HealthGPT-XL32
        await execAsync(`python app.py --task_type ct2mri --input_path ${filePath} --output_path ${outputPath}`);
        result = await fileToBase64(outputPath);
        break;
        
      case 'mri2ct':
        // MRI to CT conversion using HealthGPT-XL32
        await execAsync(`python app.py --task_type mri2ct --input_path ${filePath} --output_path ${outputPath}`);
        result = await fileToBase64(outputPath);
        break;
        
      case 'diagnosis':
        // Generate diagnosis using HealthGPT-XL32
        const { stdout } = await execAsync(`python app.py --task_type diagnosis --input_path ${filePath}`);
        result = stdout.trim();
        break;
        
      default:
        return NextResponse.json({ error: 'Invalid task type' }, { status: 400 });
    }
    
    // Clean up temporary files
    await fs.promises.unlink(filePath);
    if (taskType !== 'diagnosis') {
      await fs.promises.unlink(outputPath);
    }
    
    return NextResponse.json({ result });
  } catch (error) {
    console.error('Error processing scan:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
} 