#!/usr/bin/env npx tsx
/**
 * Generate SVG plots from benchmark results
 *
 * Usage:
 *   npx tsx benchmarks/generate-plots.ts [results.json]
 *
 * Output:
 *   benchmarks/plots/*.svg
 */

import * as fs from 'fs';
import * as path from 'path';

interface BenchmarkResult {
  name: string;
  backend: string;
  dimensions: number;
  iterations: number;
  totalTime: number;
  iterationsPerSecond: number;
  avgTimePerIteration: number;
}

interface ResultsFile {
  timestamp: string;
  results: {
    wasm: Record<string, BenchmarkResult>;
    webgpu: Record<string, BenchmarkResult>;
  };
}

// SVG generation utilities
function escapeXml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function generateBarChart(
  title: string,
  data: Array<{ label: string; wasm?: number; webgpu?: number }>,
  yAxisLabel: string,
  width = 800,
  height = 400
): string {
  const margin = { top: 60, right: 120, bottom: 80, left: 80 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  const maxValue = Math.max(
    ...data.flatMap((d) => [d.wasm ?? 0, d.webgpu ?? 0])
  );
  const yScale = chartHeight / (maxValue * 1.1);

  const barWidth = chartWidth / data.length / 3;
  const groupWidth = chartWidth / data.length;

  let bars = '';
  let labels = '';
  let legend = '';

  // Y-axis grid lines and labels
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const y = margin.top + chartHeight - (i / yTicks) * chartHeight;
    const value = ((maxValue * 1.1 * i) / yTicks).toFixed(0);
    bars += `<line x1="${margin.left}" y1="${y}" x2="${margin.left + chartWidth}" y2="${y}" stroke="#e0e0e0" stroke-width="1"/>`;
    labels += `<text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" font-size="11" fill="#666">${value}</text>`;
  }

  // Bars
  data.forEach((d, i) => {
    const x = margin.left + i * groupWidth + groupWidth / 2;

    if (d.wasm !== undefined) {
      const barHeight = d.wasm * yScale;
      const y = margin.top + chartHeight - barHeight;
      bars += `<rect x="${x - barWidth - 2}" y="${y}" width="${barWidth}" height="${barHeight}" fill="#4285f4" rx="2"/>`;
      bars += `<text x="${x - barWidth / 2 - 2}" y="${y - 5}" text-anchor="middle" font-size="10" fill="#4285f4">${d.wasm.toFixed(0)}</text>`;
    }

    if (d.webgpu !== undefined) {
      const barHeight = d.webgpu * yScale;
      const y = margin.top + chartHeight - barHeight;
      bars += `<rect x="${x + 2}" y="${y}" width="${barWidth}" height="${barHeight}" fill="#34a853" rx="2"/>`;
      bars += `<text x="${x + barWidth / 2 + 2}" y="${y - 5}" text-anchor="middle" font-size="10" fill="#34a853">${d.webgpu.toFixed(0)}</text>`;
    }

    // X-axis label
    labels += `<text x="${x}" y="${margin.top + chartHeight + 20}" text-anchor="middle" font-size="11" fill="#333">${escapeXml(d.label)}</text>`;
  });

  // Legend
  legend = `
    <rect x="${width - margin.right + 10}" y="${margin.top}" width="15" height="15" fill="#4285f4" rx="2"/>
    <text x="${width - margin.right + 30}" y="${margin.top + 12}" font-size="12" fill="#333">WASM</text>
    <rect x="${width - margin.right + 10}" y="${margin.top + 25}" width="15" height="15" fill="#34a853" rx="2"/>
    <text x="${width - margin.right + 30}" y="${margin.top + 37}" font-size="12" fill="#333">WebGPU</text>
  `;

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
  <style>
    text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
  </style>
  <rect width="${width}" height="${height}" fill="white"/>

  <!-- Title -->
  <text x="${width / 2}" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">${escapeXml(title)}</text>

  <!-- Y-axis label -->
  <text x="20" y="${height / 2}" text-anchor="middle" font-size="12" fill="#666" transform="rotate(-90, 20, ${height / 2})">${escapeXml(yAxisLabel)}</text>

  <!-- Axes -->
  <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + chartHeight}" stroke="#333" stroke-width="1"/>
  <line x1="${margin.left}" y1="${margin.top + chartHeight}" x2="${margin.left + chartWidth}" y2="${margin.top + chartHeight}" stroke="#333" stroke-width="1"/>

  ${bars}
  ${labels}
  ${legend}
</svg>`;
}

function generateSpeedupChart(
  title: string,
  data: Array<{ label: string; speedup: number }>,
  width = 800,
  height = 400
): string {
  const margin = { top: 60, right: 40, bottom: 80, left: 80 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  const maxSpeedup = Math.max(...data.map((d) => d.speedup), 2);
  const minSpeedup = Math.min(...data.map((d) => d.speedup), 0.5);
  const range = maxSpeedup - minSpeedup;
  const yScale = chartHeight / (range * 1.2);
  const baseline = margin.top + chartHeight - (1 - minSpeedup) * yScale * 1.2;

  const barWidth = chartWidth / data.length / 2;
  const groupWidth = chartWidth / data.length;

  let bars = '';
  let labels = '';

  // Baseline at 1x
  bars += `<line x1="${margin.left}" y1="${baseline}" x2="${margin.left + chartWidth}" y2="${baseline}" stroke="#ff6b6b" stroke-width="2" stroke-dasharray="5,5"/>`;
  labels += `<text x="${margin.left + chartWidth + 5}" y="${baseline + 4}" font-size="11" fill="#ff6b6b">1x (no speedup)</text>`;

  // Y-axis grid lines
  const yTicks = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5].filter(
    (v) => v >= minSpeedup * 0.8 && v <= maxSpeedup * 1.2
  );
  for (const tick of yTicks) {
    const y = margin.top + chartHeight - (tick - minSpeedup) * yScale * 1.2;
    if (y > margin.top && y < margin.top + chartHeight) {
      bars += `<line x1="${margin.left}" y1="${y}" x2="${margin.left + chartWidth}" y2="${y}" stroke="#e0e0e0" stroke-width="1"/>`;
      labels += `<text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" font-size="11" fill="#666">${tick}x</text>`;
    }
  }

  // Bars
  data.forEach((d, i) => {
    const x = margin.left + i * groupWidth + groupWidth / 2;
    const barHeight = Math.abs(d.speedup - 1) * yScale * 1.2;
    const y =
      d.speedup >= 1 ? baseline - barHeight : baseline;
    const color = d.speedup >= 1 ? '#34a853' : '#ea4335';

    bars += `<rect x="${x - barWidth / 2}" y="${y}" width="${barWidth}" height="${barHeight}" fill="${color}" rx="2"/>`;
    bars += `<text x="${x}" y="${d.speedup >= 1 ? y - 5 : y + barHeight + 15}" text-anchor="middle" font-size="11" font-weight="bold" fill="${color}">${d.speedup.toFixed(2)}x</text>`;

    // X-axis label
    labels += `<text x="${x}" y="${margin.top + chartHeight + 20}" text-anchor="middle" font-size="11" fill="#333">${escapeXml(d.label)}</text>`;
  });

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
  <style>
    text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
  </style>
  <rect width="${width}" height="${height}" fill="white"/>

  <!-- Title -->
  <text x="${width / 2}" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">${escapeXml(title)}</text>

  <!-- Y-axis label -->
  <text x="20" y="${height / 2}" text-anchor="middle" font-size="12" fill="#666" transform="rotate(-90, 20, ${height / 2})">Speedup (WebGPU / WASM)</text>

  <!-- Axes -->
  <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + chartHeight}" stroke="#333" stroke-width="1"/>
  <line x1="${margin.left}" y1="${margin.top + chartHeight}" x2="${margin.left + chartWidth}" y2="${margin.top + chartHeight}" stroke="#333" stroke-width="1"/>

  ${bars}
  ${labels}
</svg>`;
}

function generateScalingChart(
  title: string,
  data: Array<{ dimensions: number; wasm?: number; webgpu?: number }>,
  width = 800,
  height = 400
): string {
  const margin = { top: 60, right: 120, bottom: 60, left: 80 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  const maxValue = Math.max(
    ...data.flatMap((d) => [d.wasm ?? 0, d.webgpu ?? 0])
  );
  const maxDim = Math.max(...data.map((d) => d.dimensions));

  const xScale = chartWidth / Math.log10(maxDim + 1);
  const yScale = chartHeight / (maxValue * 1.1);

  let paths = '';
  let points = '';
  let labels = '';

  // Grid lines
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const y = margin.top + chartHeight - (i / yTicks) * chartHeight;
    const value = ((maxValue * 1.1 * i) / yTicks).toFixed(0);
    paths += `<line x1="${margin.left}" y1="${y}" x2="${margin.left + chartWidth}" y2="${y}" stroke="#e0e0e0" stroke-width="1"/>`;
    labels += `<text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" font-size="11" fill="#666">${value}</text>`;
  }

  // X-axis labels (log scale)
  for (const dim of [1, 10, 100]) {
    if (dim <= maxDim) {
      const x = margin.left + Math.log10(dim + 1) * xScale;
      labels += `<text x="${x}" y="${margin.top + chartHeight + 20}" text-anchor="middle" font-size="11" fill="#333">${dim}D</text>`;
    }
  }

  // WASM line
  const wasmPoints = data
    .filter((d) => d.wasm !== undefined)
    .map((d) => ({
      x: margin.left + Math.log10(d.dimensions + 1) * xScale,
      y: margin.top + chartHeight - d.wasm! * yScale,
    }));

  if (wasmPoints.length > 1) {
    const pathD = wasmPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
    paths += `<path d="${pathD}" fill="none" stroke="#4285f4" stroke-width="2"/>`;
  }
  wasmPoints.forEach((p) => {
    points += `<circle cx="${p.x}" cy="${p.y}" r="4" fill="#4285f4"/>`;
  });

  // WebGPU line
  const webgpuPoints = data
    .filter((d) => d.webgpu !== undefined)
    .map((d) => ({
      x: margin.left + Math.log10(d.dimensions + 1) * xScale,
      y: margin.top + chartHeight - d.webgpu! * yScale,
    }));

  if (webgpuPoints.length > 1) {
    const pathD = webgpuPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
    paths += `<path d="${pathD}" fill="none" stroke="#34a853" stroke-width="2"/>`;
  }
  webgpuPoints.forEach((p) => {
    points += `<circle cx="${p.x}" cy="${p.y}" r="4" fill="#34a853"/>`;
  });

  // Legend
  const legend = `
    <line x1="${width - margin.right + 10}" y1="${margin.top + 7}" x2="${width - margin.right + 25}" y2="${margin.top + 7}" stroke="#4285f4" stroke-width="2"/>
    <circle cx="${width - margin.right + 17.5}" cy="${margin.top + 7}" r="3" fill="#4285f4"/>
    <text x="${width - margin.right + 30}" y="${margin.top + 12}" font-size="12" fill="#333">WASM</text>

    <line x1="${width - margin.right + 10}" y1="${margin.top + 32}" x2="${width - margin.right + 25}" y2="${margin.top + 32}" stroke="#34a853" stroke-width="2"/>
    <circle cx="${width - margin.right + 17.5}" cy="${margin.top + 32}" r="3" fill="#34a853"/>
    <text x="${width - margin.right + 30}" y="${margin.top + 37}" font-size="12" fill="#333">WebGPU</text>
  `;

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
  <style>
    text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
  </style>
  <rect width="${width}" height="${height}" fill="white"/>

  <!-- Title -->
  <text x="${width / 2}" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">${escapeXml(title)}</text>

  <!-- Axis labels -->
  <text x="20" y="${height / 2}" text-anchor="middle" font-size="12" fill="#666" transform="rotate(-90, 20, ${height / 2})">Iterations per second</text>
  <text x="${margin.left + chartWidth / 2}" y="${height - 15}" text-anchor="middle" font-size="12" fill="#666">Dimensions (log scale)</text>

  <!-- Axes -->
  <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + chartHeight}" stroke="#333" stroke-width="1"/>
  <line x1="${margin.left}" y1="${margin.top + chartHeight}" x2="${margin.left + chartWidth}" y2="${margin.top + chartHeight}" stroke="#333" stroke-width="1"/>

  ${paths}
  ${points}
  ${labels}
  ${legend}
</svg>`;
}

// Interface for array-format results from benchmark-wasm.ts
interface ArrayResultsFile {
  backend: string;
  timestamp: string;
  platform?: string;
  nodeVersion?: string;
  results: Array<{
    config: { name: string; dimensions: number };
    backend: string;
    iterationsPerSecond: number;
    totalTime: number;
    avgTimePerIteration: number;
  }>;
}

// Convert array results to object format
function convertArrayResults(data: ArrayResultsFile): ResultsFile {
  const backend = data.backend as 'wasm' | 'webgpu';
  const byName: Record<string, BenchmarkResult> = {};

  for (const r of data.results) {
    byName[r.config.name] = {
      name: r.config.name,
      backend: r.backend,
      dimensions: r.config.dimensions,
      iterations: 0,
      totalTime: r.totalTime,
      iterationsPerSecond: r.iterationsPerSecond,
      avgTimePerIteration: r.avgTimePerIteration,
    };
  }

  return {
    timestamp: data.timestamp,
    results: {
      wasm: backend === 'wasm' ? byName : {},
      webgpu: backend === 'webgpu' ? byName : {},
    },
  };
}

async function main() {
  const inputFile = process.argv[2] ?? 'benchmarks/results-wasm.json';
  const outputDir = 'benchmarks/plots';

  // Create output directory
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Check if input file exists
  if (!fs.existsSync(inputFile)) {
    console.log(`Input file not found: ${inputFile}`);
    console.log('\nGenerating example plots with simulated data...\n');

    // Generate example data for demonstration
    const exampleData = generateExampleData();
    generatePlotsFromData(exampleData, outputDir);
    return;
  }

  // Read results file
  const rawData = JSON.parse(fs.readFileSync(inputFile, 'utf-8'));

  // Detect format and convert if necessary
  let results: ResultsFile;
  if (Array.isArray(rawData.results)) {
    // Array format from benchmark-wasm.ts
    results = convertArrayResults(rawData as ArrayResultsFile);
  } else {
    // Already in object format
    results = rawData as ResultsFile;
  }

  generatePlotsFromData(results, outputDir);
}

function generateExampleData(): ResultsFile {
  // Simulated benchmark results for demonstration
  // Based on typical WebGPU vs WASM performance characteristics
  return {
    timestamp: new Date().toISOString(),
    results: {
      wasm: {
        'HMC-1D': { name: 'HMC-1D', backend: 'wasm', dimensions: 1, iterations: 1000, totalTime: 500, iterationsPerSecond: 2000, avgTimePerIteration: 0.5 },
        'HMC-10D': { name: 'HMC-10D', backend: 'wasm', dimensions: 10, iterations: 1000, totalTime: 800, iterationsPerSecond: 1250, avgTimePerIteration: 0.8 },
        'HMC-50D': { name: 'HMC-50D', backend: 'wasm', dimensions: 50, iterations: 500, totalTime: 1200, iterationsPerSecond: 417, avgTimePerIteration: 2.4 },
        'HMC-100D': { name: 'HMC-100D', backend: 'wasm', dimensions: 100, iterations: 500, totalTime: 2500, iterationsPerSecond: 200, avgTimePerIteration: 5 },
        'RWM-10D': { name: 'RWM-10D', backend: 'wasm', dimensions: 10, iterations: 1000, totalTime: 400, iterationsPerSecond: 2500, avgTimePerIteration: 0.4 },
        'RWM-50D': { name: 'RWM-50D', backend: 'wasm', dimensions: 50, iterations: 500, totalTime: 600, iterationsPerSecond: 833, avgTimePerIteration: 1.2 },
      },
      webgpu: {
        'HMC-1D': { name: 'HMC-1D', backend: 'webgpu', dimensions: 1, iterations: 1000, totalTime: 600, iterationsPerSecond: 1667, avgTimePerIteration: 0.6 },
        'HMC-10D': { name: 'HMC-10D', backend: 'webgpu', dimensions: 10, iterations: 1000, totalTime: 500, iterationsPerSecond: 2000, avgTimePerIteration: 0.5 },
        'HMC-50D': { name: 'HMC-50D', backend: 'webgpu', dimensions: 50, iterations: 500, totalTime: 450, iterationsPerSecond: 1111, avgTimePerIteration: 0.9 },
        'HMC-100D': { name: 'HMC-100D', backend: 'webgpu', dimensions: 100, iterations: 500, totalTime: 600, iterationsPerSecond: 833, avgTimePerIteration: 1.2 },
        'RWM-10D': { name: 'RWM-10D', backend: 'webgpu', dimensions: 10, iterations: 1000, totalTime: 350, iterationsPerSecond: 2857, avgTimePerIteration: 0.35 },
        'RWM-50D': { name: 'RWM-50D', backend: 'webgpu', dimensions: 50, iterations: 500, totalTime: 300, iterationsPerSecond: 1667, avgTimePerIteration: 0.6 },
      },
    },
  };
}

function generatePlotsFromData(results: ResultsFile, outputDir: string) {
  const { wasm, webgpu } = results.results;

  // 1. Throughput comparison bar chart
  const throughputData = Object.keys({ ...wasm, ...webgpu }).map((name) => ({
    label: name,
    wasm: wasm[name]?.iterationsPerSecond,
    webgpu: webgpu[name]?.iterationsPerSecond,
  }));

  const throughputChart = generateBarChart(
    'MCMC Sampler Throughput: WebGPU vs WASM',
    throughputData,
    'Iterations per second'
  );
  fs.writeFileSync(path.join(outputDir, 'throughput-comparison.svg'), throughputChart);
  console.log('Generated: throughput-comparison.svg');

  // 2. Speedup chart
  const speedupData = Object.keys({ ...wasm, ...webgpu })
    .filter((name) => wasm[name] && webgpu[name])
    .map((name) => ({
      label: name,
      speedup: webgpu[name].iterationsPerSecond / wasm[name].iterationsPerSecond,
    }));

  if (speedupData.length > 0) {
    const speedupChart = generateSpeedupChart(
      'WebGPU Speedup over WASM',
      speedupData
    );
    fs.writeFileSync(path.join(outputDir, 'speedup.svg'), speedupChart);
    console.log('Generated: speedup.svg');
  }

  // 3. Scaling chart (if we have dimension data)
  const scalingData = Object.keys(wasm)
    .map((name) => ({
      dimensions: wasm[name]?.dimensions ?? 0,
      wasm: wasm[name]?.iterationsPerSecond,
      webgpu: webgpu[name]?.iterationsPerSecond,
    }))
    .filter((d) => d.dimensions > 0)
    .sort((a, b) => a.dimensions - b.dimensions);

  if (scalingData.length > 2) {
    const scalingChart = generateScalingChart(
      'Performance Scaling with Dimensionality',
      scalingData
    );
    fs.writeFileSync(path.join(outputDir, 'scaling.svg'), scalingChart);
    console.log('Generated: scaling.svg');
  }

  // 4. HMC-specific chart
  const hmcData = Object.keys({ ...wasm, ...webgpu })
    .filter((name) => name.includes('HMC'))
    .map((name) => ({
      label: name.replace('HMC-', '').replace('-Gaussian', ''),
      wasm: wasm[name]?.iterationsPerSecond,
      webgpu: webgpu[name]?.iterationsPerSecond,
    }));

  if (hmcData.length > 0) {
    const hmcChart = generateBarChart(
      'HMC Sampler Performance by Dimension',
      hmcData,
      'Iterations per second'
    );
    fs.writeFileSync(path.join(outputDir, 'hmc-performance.svg'), hmcChart);
    console.log('Generated: hmc-performance.svg');
  }

  console.log(`\nAll plots saved to: ${outputDir}/`);
}

main().catch(console.error);
