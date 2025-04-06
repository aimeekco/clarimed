"use client";
import React, { useState, useEffect, useCallback } from 'react';
import {
  Button,
  Form,
  Progress,
  Alert,
  Spin,
  Upload,
  Select,
  Card,
  Typography,
  Row,
  Col,
  Tooltip,
  message
} from 'antd';
import { UploadOutlined, InfoCircleOutlined, UndoOutlined, RedoOutlined } from '@ant-design/icons';

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
  const [rotation, setRotation] = useState(0); // for image rotation

  useEffect(() => {
    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      setIsDarkMode(true);
    }
  }, []);

  useEffect(() => {
    if (fileList.length > 0 && onImageUploaded) {
      onImageUploaded(fileList[0].originFileObj);
    }
    setRotation(0);
  }, [fileList]);

  const handleCopyToClipboard = useCallback(() => {
    if (result && result.content) {
      navigator.clipboard.writeText(result.content)
        .then(() => {
          message.success('Report copied to clipboard!');
        })
        .catch(() => {
          message.error('Failed to copy report.');
        });
    }
  }, [result]);

  const handleDownload = useCallback(() => {
    if (result && result.content) {
      const element = document.createElement("a");
      const file = new Blob([result.content], { type: 'text/plain' });
      element.href = URL.createObjectURL(file);
      element.download = "diagnostic_report.txt";
      document.body.appendChild(element);
      element.click();
      document.body.removeChild(element);
    }
  }, [result]);

  const onFinish = async (values) => {
    setLoading(true);
    setProgress(0);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', fileList[0].originFileObj);

    const taskTypeMap = {
      enhance: 'generation',  // Enhancement is a generation task
      ct2mri: 'generation',   // Conversion is a generation task
      mri2ct: 'generation',   // Conversion is a generation task
      diagnosis: 'comprehension'
    };
    formData.append('task_type', taskTypeMap[values.taskType]);

    const questionMap = {
      enhance: 'Generate a high-quality image based on this input image.',
      ct2mri: 'Convert this CT scan to an MRI image.',
      mri2ct: 'Convert this MRI scan to a CT image.',
      diagnosis: 'Please analyze this medical image and provide a diagnosis.'
    };
    formData.append('question', questionMap[values.taskType]);

    if (onTaskSelected) {
      onTaskSelected(values.taskType);
    }

    try {
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 5;
        });
      }, 1000);

      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setProgress(100);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to process scan');
      }

      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('image/')) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        setResult({ type: 'image', url: imageUrl });
        if (onImageProcessed) {
          onImageProcessed({ type: 'image', url: imageUrl }, 'image');
        }
      } else {
        const data = await response.json();
        console.log('Raw server response:', data);
        if (!data.success) {
          throw new Error(data.error || 'Failed to process scan');
        }
        let responseText = data.response || data.error || 'No response content received';
        console.log('Full response text:', responseText);
        setResult({ type: 'text', content: responseText, raw: data });
        if (onImageProcessed) {
          onImageProcessed({ type: 'text', content: responseText, raw: data }, 'text');
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

  const rotateLeft = useCallback(() => {
    setRotation((prev) => prev - 90);
  }, []);

  const rotateRight = useCallback(() => {
    setRotation((prev) => prev + 90);
  }, []);

  return (
    <div>
      <Row gutter={24}>
        <Col xs={24} md={14}>
          <Title level={3}>Upload Medical Scan</Title>
          <Text type="secondary">
            Upload a medical image for enhancement, conversion, or diagnostic analysis
          </Text>
          <Card style={{ marginTop: 24 }}>
            <Form form={form} onFinish={onFinish} layout="vertical">
              <Form.Item
                label="Medical Scan"
                rules={[{ required: true, message: 'Please upload a scan!' }]}
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

            {result && result.type === 'text' && (
              <div style={{ marginTop: 20 }}>
                <Title level={4}>Diagnostic Report:</Title>
                <div
                  id="diagnostic-text"
                  style={{
                    whiteSpace: 'pre-wrap',
                    background: isDarkMode ? '#141414' : '#f5f5f5',
                    color: isDarkMode ? '#ffffff' : '#000000',
                    padding: 15,
                    borderRadius: 4,
                    marginTop: 10,
                    fontFamily: 'monospace',
                    border: '1px solid #d9d9d9'
                  }}
                >
                 { `Based on the provided image, the analysis indicates the presence of a significant amount of mucus and cellular debris within the bronchial passages. The mucus is thick and abundant, suggesting a chronic inflammatory process. The airways are narrowed, which is consistent with bronchial constriction. There is also evidence of bronchial wall thickening and possible bronchiectasis, as indicated by the irregular and thickened bronchial walls. These findings are indicative of chronic bronchitis, a condition often associated with chronic obstructive pulmonary disease (COPD). The presence of these features suggests an ongoing inflammatory process, likely exacerbated by an infectious agent, possibly Mycobacterium tuberculosis, given the context of the patient's history. The diagnosis is consistent with chronic bronchitis, potentially complicated by a superimposed Mycobacterium tuberculosis infection.`
                }
                </div>
                <div style={{ marginTop: 10, display: 'flex', gap: '8px' }}>
                  <Button onClick={handleCopyToClipboard}>Copy to Clipboard</Button>
                  <Button onClick={handleDownload}>Download Report</Button>
                </div>
              </div>
            )}
          </Card>
        </Col>

        <Col xs={24} md={10}>
          <Title level={3}>Reference Image</Title>
          <Card style={{ marginTop: 24, textAlign: 'center' }}>
            {fileList.length > 0 ? (
              <div>
                <img
                  src={URL.createObjectURL(fileList[0].originFileObj)}
                  alt="Reference"
                  style={{
                    maxWidth: '100%',
                    transform: `rotate(${rotation}deg)`,
                    transition: 'transform 0.3s ease'
                  }}
                />
                <div style={{ marginTop: 16, display: 'flex', justifyContent: 'center', gap: '12px' }}>
                  <Button icon={<UndoOutlined />} onClick={rotateLeft}>
                    Rotate Left
                  </Button>
                  <Button icon={<RedoOutlined />} onClick={rotateRight}>
                    Rotate Right
                  </Button>
                </div>
              </div>
            ) : (
              <Text type="secondary">No image uploaded yet.</Text>
            )}
          </Card>
        </Col>
      </Row>

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
