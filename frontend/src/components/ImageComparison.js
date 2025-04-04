"use client";
import React, { useState, useRef, useEffect } from 'react';
import { Card, Slider, Row, Col, Typography, Empty, Tooltip, Button } from 'antd';
import { 
  ZoomInOutlined, 
  ZoomOutOutlined, 
  SwapOutlined,
  DownloadOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;

const ImageComparison = ({ originalImage, processedImage, taskType }) => {
  const [sliderValue, setSliderValue] = useState(50);
  const [zoom, setZoom] = useState(1);
  const [isSwapped, setIsSwapped] = useState(false);
  const containerRef = useRef(null);
  const [containerWidth, setContainerWidth] = useState(0);

  useEffect(() => {
    if (containerRef.current) {
      setContainerWidth(containerRef.current.offsetWidth);
    }
  }, []);

  const handleSliderChange = (value) => {
    setSliderValue(value);
  };

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 0.1, 3));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 0.1, 0.5));
  };

  const handleSwap = () => {
    setIsSwapped(!isSwapped);
  };

  const handleDownload = (imageUrl, filename) => {
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (!originalImage || !processedImage) {
    return (
      <Empty 
        description="No images to compare. Please upload and process an image first." 
        style={{ margin: '40px 0' }}
      />
    );
  }

  const leftImage = isSwapped ? processedImage : originalImage;
  const rightImage = isSwapped ? originalImage : processedImage;
  const leftLabel = isSwapped ? 'Processed' : 'Original';
  const rightLabel = isSwapped ? 'Original' : 'Processed';

  return (
    <div ref={containerRef}>
      <Title level={3}>Image Comparison</Title>
      <Text type="secondary">
        {taskType === 'enhance' ? 'Super-Resolution Enhancement' : 
         taskType === 'ct2mri' ? 'CT to MRI Conversion' : 
         taskType === 'mri2ct' ? 'MRI to CT Conversion' : 'Image Processing'}
      </Text>
      
      <div style={{ marginTop: 20, marginBottom: 20 }}>
        <Row gutter={16}>
          <Col span={12}>
            <Card 
              title={leftLabel}
              extra={
                <Tooltip title="Download">
                  <Button 
                    icon={<DownloadOutlined />} 
                    onClick={() => handleDownload(leftImage, `${leftLabel.toLowerCase()}_image.png`)}
                  />
                </Tooltip>
              }
            >
              <div style={{ 
                overflow: 'hidden', 
                position: 'relative',
                height: '400px',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center'
              }}>
                <img 
                  src={leftImage} 
                  alt={leftLabel} 
                  style={{ 
                    maxWidth: '100%', 
                    maxHeight: '100%',
                    transform: `scale(${zoom})`,
                    transition: 'transform 0.3s ease'
                  }} 
                />
              </div>
            </Card>
          </Col>
          <Col span={12}>
            <Card 
              title={rightLabel}
              extra={
                <Tooltip title="Download">
                  <Button 
                    icon={<DownloadOutlined />} 
                    onClick={() => handleDownload(rightImage, `${rightLabel.toLowerCase()}_image.png`)}
                  />
                </Tooltip>
              }
            >
              <div style={{ 
                overflow: 'hidden', 
                position: 'relative',
                height: '400px',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center'
              }}>
                <img 
                  src={rightImage} 
                  alt={rightLabel} 
                  style={{ 
                    maxWidth: '100%', 
                    maxHeight: '100%',
                    transform: `scale(${zoom})`,
                    transition: 'transform 0.3s ease'
                  }} 
                />
              </div>
            </Card>
          </Col>
        </Row>
      </div>

      <div style={{ marginTop: 20 }}>
        <Row gutter={16} align="middle">
          <Col span={2}>
            <Tooltip title="Zoom Out">
              <Button icon={<ZoomOutOutlined />} onClick={handleZoomOut} />
            </Tooltip>
          </Col>
          <Col span={18}>
            <Slider 
              value={sliderValue} 
              onChange={handleSliderChange}
              tooltip={{ formatter: null }}
            />
          </Col>
          <Col span={2}>
            <Tooltip title="Zoom In">
              <Button icon={<ZoomInOutlined />} onClick={handleZoomIn} />
            </Tooltip>
          </Col>
          <Col span={2}>
            <Tooltip title="Swap Images">
              <Button icon={<SwapOutlined />} onClick={handleSwap} />
            </Tooltip>
          </Col>
        </Row>
      </div>
    </div>
  );
};

export default ImageComparison; 