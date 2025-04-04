"use client";
import React, { useState } from 'react';
import { Card, Form, Switch, Input, Button, Select, Typography, Divider, message } from 'antd';
import { SaveOutlined, ReloadOutlined } from '@ant-design/icons';
import { useTheme } from '../context/ThemeContext';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

const Settings = () => {
  const { isDarkMode, toggleTheme } = useTheme();
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);

  const onFinish = (values) => {
    setLoading(true);
    
    // Simulate saving settings
    setTimeout(() => {
      console.log('Settings saved:', values);
      message.success('Settings saved successfully');
      setLoading(false);
    }, 1000);
  };

  const handleReset = () => {
    form.resetFields();
    message.info('Settings reset to defaults');
  };

  return (
    <div>
      <Title level={3}>Settings</Title>
      <Text type="secondary">Configure application preferences and API settings</Text>
      
      <Card style={{ marginTop: 24 }}>
        <Form
          form={form}
          layout="vertical"
          onFinish={onFinish}
          initialValues={{
            darkMode: isDarkMode,
            apiEndpoint: 'http://localhost:8000',
            imageQuality: 'high',
            autoSave: true,
            notificationEnabled: true
          }}
        >
          <Title level={4}>Appearance</Title>
          <Form.Item
            name="darkMode"
            label="Dark Mode"
            valuePropName="checked"
          >
            <Switch 
              checked={isDarkMode} 
              onChange={toggleTheme}
              checkedChildren="On" 
              unCheckedChildren="Off" 
            />
          </Form.Item>
          
          <Divider />
          
          <Title level={4}>API Configuration</Title>
          <Form.Item
            name="apiEndpoint"
            label="API Endpoint"
            rules={[{ required: true, message: 'Please enter the API endpoint' }]}
          >
            <Input placeholder="http://localhost:8000" />
          </Form.Item>
          
          <Form.Item
            name="apiKey"
            label="API Key (Optional)"
          >
            <Input.Password placeholder="Enter your API key" />
          </Form.Item>
          
          <Divider />
          
          <Title level={4}>Processing Options</Title>
          <Form.Item
            name="imageQuality"
            label="Image Quality"
          >
            <Select>
              <Option value="low">Low (Faster)</Option>
              <Option value="medium">Medium (Balanced)</Option>
              <Option value="high">High (Better Quality)</Option>
            </Select>
          </Form.Item>
          
          <Form.Item
            name="autoSave"
            label="Auto-save Results"
            valuePropName="checked"
          >
            <Switch checkedChildren="On" unCheckedChildren="Off" />
          </Form.Item>
          
          <Form.Item
            name="notificationEnabled"
            label="Enable Notifications"
            valuePropName="checked"
          >
            <Switch checkedChildren="On" unCheckedChildren="Off" />
          </Form.Item>
          
          <Divider />
          
          <Form.Item>
            <Button 
              type="primary" 
              htmlType="submit" 
              icon={<SaveOutlined />} 
              loading={loading}
              style={{ marginRight: 8 }}
            >
              Save Settings
            </Button>
            <Button 
              icon={<ReloadOutlined />} 
              onClick={handleReset}
            >
              Reset to Defaults
            </Button>
          </Form.Item>
        </Form>
      </Card>
      
      <Card style={{ marginTop: 24 }}>
        <Title level={4}>About</Title>
        <Paragraph>
          HealthGPT Medical Imaging Suite v1.0.0
        </Paragraph>
        <Paragraph>
          This application uses advanced AI models to enhance medical images and generate diagnostic reports.
          It is designed for use in low-resource healthcare settings where high-end imaging equipment may not be available.
        </Paragraph>
      </Card>
    </div>
  );
};

export default Settings; 