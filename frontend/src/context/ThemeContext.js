"use client";
import { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    console.log('Initial theme check');
    // Check if user has a theme preference in localStorage
    const savedTheme = localStorage.getItem('theme');
    console.log('Saved theme:', savedTheme);
    if (savedTheme) {
      setIsDarkMode(savedTheme === 'dark');
    } else {
      // Check system preference
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      console.log('System prefers dark:', prefersDark);
      setIsDarkMode(prefersDark);
    }
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;
    
    console.log('Theme changed to:', isDarkMode ? 'dark' : 'light');
    // Update localStorage and document class when theme changes
    localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
    const html = document.documentElement;
    console.log('HTML class before:', html.className);
    if (isDarkMode) {
      html.classList.add('dark');
    } else {
      html.classList.remove('dark');
    }
    console.log('HTML class after:', html.className);
  }, [isDarkMode, mounted]);

  const toggleTheme = () => {
    console.log('Toggle theme clicked, current state:', isDarkMode);
    setIsDarkMode(!isDarkMode);
  };

  if (!mounted) {
    return null;
  }

  return (
    <ThemeContext.Provider value={{ isDarkMode, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
} 