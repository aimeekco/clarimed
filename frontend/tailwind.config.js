/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
      "./src/**/*.{js,ts,jsx,tsx}",
    ],
    darkMode: 'class',
    theme: {
      extend: {
        colors: {
          dark: {
            bg: '#0a0a0a',
            text: '#ededed',
            border: '#333333',
          },
        },
      },
    },
    plugins: [],
  }
  