"use client";
import React, { useState } from 'react';
import { Tabs, Layout, Menu, Button, Tooltip } from 'antd';
import { 
  FileImageOutlined, 
  ExperimentOutlined, 
  FileTextOutlined, 
  SettingOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined
} from '@ant-design/icons';
import UploadForm from './UploadForm';
import ImageComparison from './ImageComparison';
import DiagnosticReport from './DiagnosticReport';
import Settings from './Settings';

const { Header, Sider, Content } = Layout;

const Dashboard = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState('1');
  const [uploadedImage, setUploadedImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [diagnosticReport, setDiagnosticReport] = useState(null);
  const [taskType, setTaskType] = useState(null);

  const handleImageProcessed = (result, type) => {
    if (type === 'image') {
      setProcessedImage(result.url);
    } else if (type === 'text') {
      setDiagnosticReport(result.content);
    }
  };

  const handleImageUploaded = (file) => {
    setUploadedImage(URL.createObjectURL(file));
  };

  const handleTaskSelected = (task) => {
    setTaskType(task);
  };

  const items = [
    {
      key: '1',
      icon: <FileImageOutlined />,
      label: 'Image Processing',
    },
    {
      key: '2',
      icon: <ExperimentOutlined />,
      label: 'Comparison',
    },
    {
      key: '3',
      icon: <FileTextOutlined />,
      label: 'Diagnostic Report',
    },
    {
      key: '4',
      icon: <SettingOutlined />,
      label: 'Settings',
    },
  ];

  const renderContent = () => {
    switch (activeTab) {
      case '1':
        return (
          <UploadForm 
            onImageProcessed={handleImageProcessed}
            onImageUploaded={handleImageUploaded}
            onTaskSelected={handleTaskSelected}
          />
        );
      case '2':
        return (
          <ImageComparison 
            originalImage={uploadedImage}
            processedImage={processedImage}
            taskType={taskType}
          />
        );
      case '3':
        return (
          <DiagnosticReport 
            report={diagnosticReport}
            originalImage={uploadedImage}
            processedImage={processedImage}
          />
        );
      case '4':
        return <Settings />;
      default:
        return null;
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider 
        trigger={null} 
        collapsible 
        collapsed={collapsed}
        theme="light"
        style={{
          overflow: 'auto',
          height: '100vh',
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0,
        }}
      >
        <div className="logo" style={{ height: 32, margin: 16, background: 'rgba(0, 0, 0, 0.2)' }} />
        <Menu
          theme="light"
          mode="inline"
          selectedKeys={[activeTab]}
          items={items}
          onClick={({ key }) => setActiveTab(key)}
        />
      </Sider>
      <Layout style={{ marginLeft: collapsed ? 80 : 200, transition: 'all 0.2s' }}>
        <Header style={{ padding: 0, background: '#fff' }}>
          <Tooltip title={collapsed ? 'Expand menu' : 'Collapse menu'}>
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed(!collapsed)}
              style={{
                fontSize: '16px',
                width: 64,
                height: 64,
              }}
            />
          </Tooltip>
        </Header>
        <Content style={{ margin: '24px 16px', padding: 24, background: '#fff', minHeight: 280 }}>
          {renderContent()}
        </Content>
      </Layout>
    </Layout>
  );
};

export default Dashboard; 