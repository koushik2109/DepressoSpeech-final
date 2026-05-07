/**
 * QualityAnalyzer - Real-time assessment of recording quality metrics
 * Analyzes: brightness, blur, motion, face visibility, and environmental conditions
 */

class QualityAnalyzer {
  constructor() {
    this.frameHistory = [];
    this.maxHistoryLength = 30; // ~1 second at 30fps
    this.thresholds = {
      minBrightness: 50,
      maxBrightness: 200,
      minBlur: 500, // Laplacian variance threshold
      maxMotion: 5, // pixels per frame
      minFaceVisibility: 0.9, // 90% landmarks visible
      centeringMargin: 0.3, // 30-70% of frame
    };
  }

  /**
   * Analyze video frame for quality metrics
   * @param {HTMLCanvasElement} canvas - Canvas with current frame
   * @param {object} faceGeometry - Face detection results
   * @returns {object} Quality metrics
   */
  analyzeFrame(canvas, faceGeometry, detectionResult = null) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Calculate quality scores
    const brightness = this.calculateBrightness(imageData);
    const blur = this.calculateBlur(imageData, canvas.width, canvas.height);
    const motion = this.calculateMotion(faceGeometry);
    const visibility = this.calculateFaceVisibility(faceGeometry);
    const centering = this.calculateFaceCentering(faceGeometry, canvas);
    const lighting = this.analyzeLighting(imageData);
    const eyeVisibility = faceGeometry?.eyesVisible ? 1.0 : 0.0;
    const faceAreaScore = this.calculateFaceAreaScore(faceGeometry);
    const multiFacePenalty = detectionResult?.multiFace ? 0 : 1;

    // Store frame history for trend analysis
    const frameMetrics = {
      timestamp: performance.now(),
      brightness,
      blur,
      motion,
      visibility,
      centering,
      lighting,
      eyeVisibility,
      faceAreaScore,
      multiFacePenalty,
    };

    this.frameHistory.push(frameMetrics);
    if (this.frameHistory.length > this.maxHistoryLength) {
      this.frameHistory.shift();
    }

    // Calculate composite quality score
    const qualityScore = this.calculateCompositeScore(frameMetrics);

    // Determine issues
    const issues = this.identifyIssues(frameMetrics, detectionResult);

    return {
      frameMetrics,
      qualityScore,
      issues,
      trend: this.calculateTrend(),
    };
  }

  /**
   * Calculate average brightness (0-255)
   */
  calculateBrightness(imageData) {
    const data = imageData.data;
    let sum = 0;

    // Process every 4th pixel for performance
    for (let i = 0; i < data.length; i += 16) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      // Luminance formula
      sum += 0.299 * r + 0.587 * g + 0.114 * b;
    }

    return sum / (data.length / 16);
  }

  /**
   * Detect blur using Laplacian variance
   * Lower variance = more blurred
   */
  calculateBlur(imageData, width, height) {
    const data = imageData.data;
    let variance = 0;
    let mean = 0;

    // Simplified Laplacian kernel application
    const step = 4; // Sample every nth pixel for performance
    let count = 0;

    for (let i = 0; i < data.length; i += step * 4) {
      const gray =
        0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      mean += gray;
      count++;
    }

    mean /= count;

    // Calculate variance
    for (let i = 0; i < data.length; i += step * 4) {
      const gray =
        0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      variance += Math.pow(gray - mean, 2);
    }

    variance /= count;
    return variance;
  }

  /**
   * Calculate motion between frames
   */
  calculateMotion(faceGeometry) {
    if (this.frameHistory.length === 0) return 0;

    const lastFrame = this.frameHistory[this.frameHistory.length - 1];
    if (!lastFrame || !lastFrame.faceGeometry) return 0;

    const dx = faceGeometry.center.x - lastFrame.faceGeometry.center.x;
    const dy = faceGeometry.center.y - lastFrame.faceGeometry.center.y;

    return Math.sqrt(dx * dx + dy * dy);
  }

  /**
   * Calculate face visibility (0-1)
   * Based on number of landmarks detected
   */
  calculateFaceVisibility(faceGeometry) {
    if (!faceGeometry) return 0;

    // Estimate visibility based on bounding box coverage and landmark spread
    const bbox = faceGeometry.boundingBox;
    const width = bbox.maxX - bbox.minX;
    const height = bbox.maxY - bbox.minY;
    const area = width * height;

    // If face is too small or landmark spread is too large, visibility is lower
    return Math.min(Math.max(area * 100, 0), 1);
  }

  /**
   * Calculate face area score (0-1)
   * Penalize when face is too small or too large in frame
   */
  calculateFaceAreaScore(faceGeometry) {
    if (!faceGeometry || !faceGeometry.faceAreaRatio) return 0;
    const ratio = faceGeometry.faceAreaRatio;
    if (ratio < 0.03) return Math.max(0, ratio / 0.03);
    if (ratio > 0.85) return Math.max(0, 1 - (ratio - 0.85) / 0.15);
    // Optimal range: 0.05 to 0.4
    if (ratio >= 0.05 && ratio <= 0.4) return 1.0;
    return 0.8;
  }

  /**
   * Calculate face centering score (0-1)
   * 1.0 when face center is in optimal range (30-70% of frame)
   */
  calculateFaceCentering(faceGeometry, canvas) {
    const faceX = faceGeometry.center.x;
    const faceY = faceGeometry.center.y;

    const xInRange =
      faceX >= canvas.width * 0.3 && faceX <= canvas.width * 0.7;
    const yInRange =
      faceY >= canvas.height * 0.25 && faceY <= canvas.height * 0.75;

    if (!xInRange || !yInRange) {
      // Calculate distance from optimal zone
      const xDist = Math.max(
        0,
        canvas.width * 0.3 - faceX,
        faceX - canvas.width * 0.7
      );
      const yDist = Math.max(
        0,
        canvas.height * 0.25 - faceY,
        faceY - canvas.height * 0.75
      );
      const totalDist = Math.sqrt(xDist * xDist + yDist * yDist);

      return Math.max(0, 1 - totalDist / (canvas.width * 0.2));
    }

    return 1.0;
  }

  /**
   * Analyze lighting conditions
   */
  analyzeLighting(imageData) {
    const data = imageData.data;
    let brightPixels = 0;
    let darkPixels = 0;

    for (let i = 0; i < data.length; i += 16) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const luminance = 0.299 * r + 0.587 * g + 0.114 * b;

      if (luminance > 200) brightPixels++;
      if (luminance < 50) darkPixels++;
    }

    const totalPixels = data.length / 16;
    const brightRatio = brightPixels / totalPixels;
    const darkRatio = darkPixels / totalPixels;

    return {
      brightRatio,
      darkRatio,
      wellLit: brightRatio < 0.3 && darkRatio < 0.3,
    };
  }

  /**
   * Calculate composite quality score (0-100)
   */
  calculateCompositeScore(frameMetrics) {
    // Multi-face penalty zeroes out the entire score
    if (frameMetrics.multiFacePenalty === 0) return 0;

    const scores = {
      brightness:
        this.scoreMetric(
          frameMetrics.brightness,
          this.thresholds.minBrightness,
          this.thresholds.maxBrightness
        ) * 0.12,
      blur:
        this.scoreMetricInverse(frameMetrics.blur, this.thresholds.minBlur) *
        0.15,
      motion:
        this.scoreMetricInverse(frameMetrics.motion, this.thresholds.maxMotion) *
        0.13,
      visibility: frameMetrics.visibility * 0.15,
      centering: frameMetrics.centering * 0.15,
      eyeVisibility: (frameMetrics.eyeVisibility || 0) * 0.15,
      faceArea: (frameMetrics.faceAreaScore || 0) * 0.15,
    };

    const totalScore = Object.values(scores).reduce((a, b) => a + b, 0);
    return Math.min(Math.max(Math.round(totalScore * 100), 0), 100);
  }

  /**
   * Score metric where optimal is between min and max
   */
  scoreMetric(value, min, max) {
    if (value < min) {
      return Math.max(0, 1 - (min - value) / min);
    }
    if (value > max) {
      return Math.max(0, 1 - (value - max) / max);
    }
    return 1.0;
  }

  /**
   * Score metric where higher is better (inverse)
   */
  scoreMetricInverse(value, threshold) {
    if (value === 0) return 1.0;
    return Math.max(0, 1 - value / threshold);
  }

  /**
   * Identify specific quality issues
   */
  identifyIssues(frameMetrics, detectionResult = null) {
    const issues = [];

    if (detectionResult?.multiFace) {
      issues.push({
        type: 'MULTI_FACE',
        severity: 'error',
        message: 'Multiple faces detected. Only one person should be visible.',
      });
    }

    if (frameMetrics.brightness < this.thresholds.minBrightness) {
      issues.push({
        type: 'TOO_DARK',
        severity: 'warning',
        message: 'Lighting is too dark. Please move to brighter location.',
      });
    }

    if (frameMetrics.brightness > this.thresholds.maxBrightness) {
      issues.push({
        type: 'TOO_BRIGHT',
        severity: 'warning',
        message: 'Lighting is too bright. Move away from direct light.',
      });
    }

    if (frameMetrics.blur < this.thresholds.minBlur) {
      issues.push({
        type: 'MOTION_BLUR',
        severity: 'warning',
        message: 'Camera motion detected. Please keep still.',
      });
    }

    if (frameMetrics.motion > this.thresholds.maxMotion) {
      issues.push({
        type: 'HEAD_MOTION',
        severity: 'warning',
        message: 'Please minimize head movement.',
      });
    }

    if (frameMetrics.visibility < this.thresholds.minFaceVisibility) {
      issues.push({
        type: 'LOW_VISIBILITY',
        severity: 'error',
        message: 'Face partially occluded. Clear face from obstructions.',
      });
    }

    if (frameMetrics.centering < 0.7) {
      issues.push({
        type: 'POOR_CENTERING',
        severity: 'info',
        message: 'Please center your face in the frame.',
      });
    }

    if (frameMetrics.eyeVisibility === 0) {
      issues.push({
        type: 'EYES_NOT_VISIBLE',
        severity: 'warning',
        message: 'Both eyes must be visible. Remove obstructions.',
      });
    }

    if (frameMetrics.faceAreaScore !== undefined && frameMetrics.faceAreaScore < 0.5) {
      issues.push({
        type: 'FACE_SIZE',
        severity: 'info',
        message: 'Adjust your distance from the camera.',
      });
    }

    return issues;
  }

  /**
   * Calculate quality trend
   */
  calculateTrend() {
    if (this.frameHistory.length < 5) return 'STABILIZING';

    const recentScores = this.frameHistory
      .slice(-5)
      .map((f) => f.qualityScore || 0);
    const avgRecent =
      recentScores.reduce((a, b) => a + b, 0) / recentScores.length;

    const olderScores = this.frameHistory
      .slice(-10, -5)
      .map((f) => f.qualityScore || 0);
    const avgOlder =
      olderScores.length > 0
        ? olderScores.reduce((a, b) => a + b, 0) / olderScores.length
        : avgRecent;

    if (avgRecent > avgOlder + 5) return 'IMPROVING';
    if (avgRecent < avgOlder - 5) return 'DEGRADING';
    return 'STABLE';
  }

  /**
   * Reset history
   */
  reset() {
    this.frameHistory = [];
  }
}

export default QualityAnalyzer;
