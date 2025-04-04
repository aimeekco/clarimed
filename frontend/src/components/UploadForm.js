"use client";
import React, { useState, useEffect } from 'react';
import { Button, Form, Progress, Alert, Spin, Upload, Select, Card, Typography, Row, Col, Tooltip } from 'antd';
import { UploadOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { ConfigProvider, theme } from 'antd';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

const UploadForm = ({ onImageProcessed, onImageUploaded, onTaskSelected }) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [fileList, setFileList] = useState([]);
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    // Check system preference on mount
    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      setIsDarkMode(true);
    }
  }, []);

  const onFinish = async (values) => {
    setLoading(true);
    setProgress(0);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('scan', fileList[0].originFileObj);
    formData.append('taskType', values.taskType);

    // Notify parent component about the selected task
    if (onTaskSelected) {
      onTaskSelected(values.taskType);
    }

    // Notify parent component about the uploaded image
    if (onImageUploaded && fileList[0]) {
      onImageUploaded(fileList[0].originFileObj);
    }

    try {
      // Start progress simulation
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 5000);

      const response = await fetch('http://localhost:8000/api/process-scan', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to process scan');
      }

      setProgress(100);

      // Handle different response types
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('image/')) {
        // For image responses (enhance, ct2mri, mri2ct)
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        setResult({ type: 'image', url: imageUrl });
        
        // Notify parent component about the processed image
        if (onImageProcessed) {
          onImageProcessed({ type: 'image', url: imageUrl }, 'image');
        }
      } else {
        // For text responses (diagnosis)
        const data = await response.json();
        setResult({ type: 'text', content: data.result });
        
        // Notify parent component about the diagnostic report
        if (onImageProcessed) {
          onImageProcessed({ type: 'text', content: data.result }, 'text');
        }
      }
    } catch (err) {
      setError(err.message);
      setProgress(0);
    } finally {
      setLoading(false);
    }
  };

  const handleChange = ({ fileList: newFileList }) => {
    setFileList(newFileList);
  };

  return (
    <div>
      <Title level={3}>Upload Medical Scan</Title>
      <Text type="secondary">Upload a medical image for enhancement, conversion, or diagnostic analysis</Text>
      
      <Card style={{ marginTop: 24 }}>
        <Form
          form={form}
          onFinish={onFinish}
          layout="vertical"
        >
          <Form.Item
            name="scan"
            label="Medical Scan"
            rules={[{ required: true, message: 'Please upload a scan!' }]}
            valuePropName="fileList"
            getValueFromEvent={(e) => {
              if (Array.isArray(e)) {
                return e;
              }
              return e?.fileList;
            }}
            extra={
              <Text type="secondary">
                Supported formats: JPG, PNG, DICOM. Max file size: 10MB.
              </Text>
            }
          >
            <Upload
              accept="image/*"
              maxCount={1}
              fileList={fileList}
              onChange={handleChange}
              beforeUpload={() => false}
              listType="picture"
            >
              <Button icon={<UploadOutlined />}>Select File</Button>
            </Upload>
          </Form.Item>

          <Form.Item
            name="taskType"
            label={
              <span>
                Task Type
                <Tooltip title="Select the type of processing to apply to your image">
                  <InfoCircleOutlined style={{ marginLeft: 8 }} />
                </Tooltip>
              </span>
            }
            rules={[{ required: true, message: 'Please select a task type!' }]}
          >
            <Select>
              <Option value="enhance">Super-Resolution Enhancement</Option>
              <Option value="ct2mri">CT to MRI Conversion</Option>
              <Option value="mri2ct">MRI to CT Conversion</Option>
              <Option value="diagnosis">Diagnostic Analysis</Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit" loading={loading} block>
              Process Scan
            </Button>
          </Form.Item>
        </Form>

        {loading && (
          <div style={{ marginTop: 20 }}>
            <Progress percent={progress} status="active" />
            <div style={{ textAlign: 'center', marginTop: 10 }}>
              <Spin /> Processing your scan...
            </div>
          </div>
        )}

        {error && (
          <Alert
            message="Error"
            description={error}
            type="error"
            showIcon
            style={{ marginTop: 20 }}
          />
        )}

        {result && (
          <div style={{ marginTop: 20 }}>
            {result.type === 'image' ? (
              <div>
                <Title level={4}>Processed Image:</Title>
                <img 
                  src={result.url} 
                  alt="Processed scan" 
                  style={{ maxWidth: '100%', marginTop: 10 }}
                />
              </div>
            ) : (
              <div>
                <Title level={4}>Diagnostic Report:</Title>
                <pre style={{ 
                  whiteSpace: 'pre-wrap',
                  background: isDarkMode ? '#141414' : '#f5f5f5',
                  color: isDarkMode ? '#ffffff' : '#000000',
                  padding: 15,
                  borderRadius: 4,
                  marginTop: 10
                }}>
                  {result.content}
                </pre>
              </div>
            )}
          </div>
        )}
      </Card>
      
      <Card style={{ marginTop: 24 }}>
        <Title level={4}>About the Tasks</Title>
        <Row gutter={[16, 16]}>
          <Col xs={24} md={12}>
            <Card size="small" title="Super-Resolution Enhancement">
              <Paragraph>
                Improves the resolution and clarity of low-quality medical images, making fine details more visible.
              </Paragraph>
            </Card>
          </Col>
          <Col xs={24} md={12}>
            <Card size="small" title="Modality Conversion">
              <Paragraph>
                Converts images between different imaging modalities (CT to MRI or MRI to CT) to simulate high-fidelity imaging.
              </Paragraph>
            </Card>
          </Col>
          <Col xs={24}>
            <Card size="small" title="Diagnostic Analysis">
              <Paragraph>
                Analyzes the medical image to generate a detailed diagnostic report with findings, interpretations, and recommendations.
              </Paragraph>
            </Card>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default UploadForm;
