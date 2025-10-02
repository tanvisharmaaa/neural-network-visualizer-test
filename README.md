# Neural Network Visualizer

An interactive web application for visualizing neural network training with real-time animations of forward and backward propagation.

## Features

- Interactive neural network visualization
- Real-time training animations
- Support for different activation functions (sigmoid, ReLU, tanh)
- Classification and regression problem types
- CSV and JSON dataset upload
- Training metrics visualization
- Adjustable animation speed and zoom

## Live Demo

The application is deployed on GitHub Pages: [https://yourusername.github.io/neural-visualizer/](https://yourusername.github.io/neural-visualizer/)

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neural-visualizer.git
cd neural-visualizer
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open [http://localhost:5173](http://localhost:5173) in your browser.

## Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Deployment

This project is automatically deployed to GitHub Pages using GitHub Actions. The deployment workflow:

1. Triggers on pushes to the `main` branch
2. Builds the project using `npm run build`
3. Deploys the built files to GitHub Pages

### Manual Deployment Setup

If you need to set up GitHub Pages manually:

1. Go to your repository settings
2. Navigate to "Pages" in the sidebar
3. Under "Source", select "GitHub Actions"
4. The workflow will automatically deploy on the next push to `main`

## Technologies Used

- React 19
- TypeScript
- TensorFlow.js
- Vite
- Chart.js
- Papa Parse

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).