"use client";
import React, { useState } from 'react';
import { Card, Row, Col, Typography, Empty, Tabs, Button, Tooltip, Divider } from 'antd';
import { 
  FileTextOutlined, 
  DownloadOutlined, 
  PrinterOutlined,
  HighlightOutlined
} from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

const DiagnosticReport = ({ report, originalImage, processedImage }) => {
  const [activeTab, setActiveTab] = useState('1');
  const [highlightedRegions, setHighlightedRegions] = useState([]);

  const handlePrint = () => {
    window.print();
  };

  const handleDownload = () => {
    const element = document.createElement('a');
    const file = new Blob([report], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = 'diagnostic_report.txt';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  if (!report) {
    return (
      <Empty 
        description="No diagnostic report available. Please process an image with diagnostic analysis." 
        style={{ margin: '40px 0' }}
      />
    );
  }

  // Function to extract sections from the report
  const extractSections = (reportText) => {
    const sections = {
      summary: '',
      findings: '',
      interpretation: '',
      recommendations: ''
    };

    // Simple parsing logic - can be enhanced based on actual report format
    const lines = reportText.split('\n');
    let currentSection = '';

    lines.forEach(line => {
      const lowerLine = line.toLowerCase();
      
      if (lowerLine.includes('summary') || lowerLine.includes('overview')) {
        currentSection = 'summary';
      } else if (lowerLine.includes('findings') || lowerLine.includes('observations')) {
        currentSection = 'findings';
      } else if (lowerLine.includes('interpretation') || lowerLine.includes('analysis')) {
        currentSection = 'interpretation';
      } else if (lowerLine.includes('recommendations') || lowerLine.includes('suggestions')) {
        currentSection = 'recommendations';
      } else if (currentSection && line.trim()) {
        sections[currentSection] += line + '\n';
      }
    });

    return sections;
  };

  const sections = extractSections(report);

  return (
    <div>
      <Title level={3}>Diagnostic Report</Title>
      
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          <Card>
            <Tabs activeKey={activeTab} onChange={setActiveTab}>
              <TabPane 
                tab={
                  <span>
                    <FileTextOutlined />
                    Report
                  </span>
                } 
                key="1"
              >
                <div className="report-content">
                  {sections.summary && (
                    <div className="report-section">
                      <Title level={4}>Summary</Title>
                      <Paragraph>
                        <ReactMarkdown>{sections.summary}</ReactMarkdown>
                      </Paragraph>
                    </div>
                  )}
                  
                  {sections.findings && (
                    <div className="report-section">
                      <Title level={4}>Findings</Title>
                      <Paragraph>
                        <ReactMarkdown>{sections.findings}</ReactMarkdown>
                      </Paragraph>
                    </div>
                  )}
                  
                  {sections.interpretation && (
                    <div className="report-section">
                      <Title level={4}>Interpretation</Title>
                      <Paragraph>
                        <ReactMarkdown>{sections.interpretation}</ReactMarkdown>
                      </Paragraph>
                    </div>
                  )}
                  
                  {sections.recommendations && (
                    <div className="report-section">
                      <Title level={4}>Recommendations</Title>
                      <Paragraph>
                        <ReactMarkdown>{sections.recommendations}</ReactMarkdown>
                      </Paragraph>
                    </div>
                  )}
                </div>
              </TabPane>
              
              <TabPane 
                tab={
                  <span>
                    <HighlightOutlined />
                    Annotations
                  </span>
                } 
                key="2"
              >
                <Empty description="Image annotations will be displayed here" />
              </TabPane>
            </Tabs>
            
            <Divider />
            
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
              <Tooltip title="Download Report">
                <Button icon={<DownloadOutlined />} onClick={handleDownload}>
                  Download
                </Button>
              </Tooltip>
              <Tooltip title="Print Report">
                <Button icon={<PrinterOutlined />} onClick={handlePrint}>
                  Print
                </Button>
              </Tooltip>
            </div>
          </Card>
        </Col>
        
        <Col xs={24} lg={8}>
          <Card title="Reference Images">
            {originalImage && (
              <div style={{ marginBottom: 16 }}>
                <Text strong>Original Image:</Text>
                <img 
                  src={originalImage} 
                  alt="Original" 
                  style={{ width: '100%', marginTop: 8 }} 
                />
              </div>
            )}
            
            {processedImage && (
              <div>
                <Text strong>Processed Image:</Text>
                <img 
                  src={processedImage} 
                  alt="Processed" 
                  style={{ width: '100%', marginTop: 8 }} 
                />
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default DiagnosticReport; 