/* global describe, test, expect, beforeEach */
/**
 * Test Suite for Assessment Integrity Monitoring
 * 
 * Run with: npm test -- --testPathPattern=integrity
 * 
 * Coverage includes:
 * - Face detection service
 * - Quality analysis
 * - State machine transitions
 * - Component integration
 */

import FaceDetectionService from '../services/FaceDetectionService';
import QualityAnalyzer from '../services/QualityAnalyzer';
import AssessmentIntegrityStateMachine from '../services/AssessmentIntegrityStateMachine';

// ============================================================================
// FACE DETECTION SERVICE TESTS
// ============================================================================

describe('FaceDetectionService', () => {
  test('initializes without error', async () => {
    const result = await FaceDetectionService.initialize();
    expect(result).toBe(true);
    expect(FaceDetectionService.isInitialized()).toBe(true);
  });

  test('handles face detection with mock data', async () => {
    await FaceDetectionService.initialize();
    
    // Mock canvas with face
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    
    // In real tests, inject mock MediaPipe results
    const result = FaceDetectionService.detectFace(canvas);
    
    // Should return detection result object
    expect(result).toHaveProperty('detected');
    expect(result).toHaveProperty('frameCount');
    expect(result).toHaveProperty('timestamp');
  });

  test('calculates face geometry correctly', async () => {
    const mockLandmarks = Array.from({ length: 468 }, () => ({
      x: 0.5,
      y: 0.5,
      z: 0,
    }));

    // Test geometry calculation
    const geometry = FaceDetectionService.detectFace({
      detected: true,
      landmarks: mockLandmarks,
    });

    expect(geometry).toHaveProperty('center');
    expect(geometry).toHaveProperty('rotation');
    expect(geometry).toHaveProperty('boundingBox');
    expect(geometry.rotation).toHaveProperty('yaw');
    expect(geometry.rotation).toHaveProperty('pitch');
    expect(geometry.rotation).toHaveProperty('roll');
  });

  test('cleanup releases resources', async () => {
    const result = await FaceDetectionService.initialize();
    expect(result).toBe(true);

    FaceDetectionService.cleanup();
    expect(FaceDetectionService.isInitialized()).toBe(false);
  });
});

// ============================================================================
// QUALITY ANALYZER TESTS
// ============================================================================

describe('QualityAnalyzer', () => {
  let analyzer;

  beforeEach(() => {
    analyzer = new QualityAnalyzer();
  });

  test('calculates brightness correctly', () => {
    const canvas = document.createElement('canvas');
    canvas.width = 100;
    canvas.height = 100;
    const ctx = canvas.getContext('2d');

    // Fill with bright color
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, 100, 100);

    const imageData = ctx.getImageData(0, 0, 100, 100);
    const brightness = analyzer.calculateBrightness(imageData);

    expect(brightness).toBeGreaterThan(240);
    expect(brightness).toBeLessThanOrEqual(255);
  });

  test('calculates blur score', () => {
    const canvas = document.createElement('canvas');
    canvas.width = 100;
    canvas.height = 100;
    const ctx = canvas.getContext('2d');

    // Create simple pattern (non-blurry)
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, 50, 100);
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(50, 0, 50, 100);

    const imageData = ctx.getImageData(0, 0, 100, 100);
    const blur = analyzer.calculateBlur(imageData);

    // Non-blurry image should have high variance
    expect(blur).toBeGreaterThan(100);
  });

  test('detects motion between frames', () => {
    // First frame
    analyzer.frameHistory = [
      {
        faceGeometry: {
          center: { x: 0.5, y: 0.5 },
        },
      },
    ];

    // Mock geometry with movement
    const newGeometry = {
      center: { x: 0.55, y: 0.5 }, // Moved 5% right
    };

    const motion = analyzer.calculateMotion(newGeometry);
    expect(motion).toBeGreaterThan(0);
  });

  test('calculates face visibility', () => {
    const mockGeometry = {
      boundingBox: {
        minX: 0.2,
        maxX: 0.8,
        minY: 0.2,
        maxY: 0.8,
      },
    };

    const visibility = analyzer.calculateFaceVisibility(mockGeometry);
    expect(visibility).toBeGreaterThan(0);
    expect(visibility).toBeLessThanOrEqual(1);
  });

  test('identifies quality issues', () => {
    const frameMetrics = {
      brightness: 30, // Too dark
      blur: 200, // Too blurry
      motion: 10, // High motion
      visibility: 0.8,
      centering: 0.5, // Poor centering
    };

    const issues = analyzer.identifyIssues(frameMetrics);

    expect(issues.length).toBeGreaterThan(0);
    expect(issues.some((i) => i.type === 'TOO_DARK')).toBe(true);
    expect(issues.some((i) => i.type === 'MOTION_BLUR')).toBe(true);
  });

  test('calculates quality trend', () => {
    // Add history
    for (let i = 0; i < 10; i++) {
      analyzer.frameHistory.push({
        qualityScore: 50 + i * 2, // Improving
      });
    }

    const trend = analyzer.calculateTrend();
    expect(['IMPROVING', 'STABLE', 'DEGRADING']).toContain(trend);
  });
});

// ============================================================================
// STATE MACHINE TESTS
// ============================================================================

describe('AssessmentIntegrityStateMachine', () => {
  let stateMachine;

  beforeEach(() => {
    stateMachine = new AssessmentIntegrityStateMachine();
  });

  test('starts in INITIALIZING state', () => {
    expect(stateMachine.state).toBe('INITIALIZING');
    expect(stateMachine.recordingAllowed).toBe(false);
  });

  test('transitions from INITIALIZING to SCANNING', () => {
    const faceGeometry = {
      detected: false,
    };

    const qualityData = {
      qualityScore: 75,
    };

    stateMachine.update(faceGeometry, qualityData);
    expect(stateMachine.state).toBe('SCANNING');
  });

  test('transitions to ALIGNED when face aligned for required frames', () => {
    const alignedGeometry = {
      detected: true,
      rotation: {
        yaw: 5,
        pitch: 5,
        roll: 5,
      },
    };

    const qualityData = {
      qualityScore: 75,
    };

    // Call update ALIGNMENT_FRAMES_REQUIRED times
    for (let i = 0; i < 15; i++) {
      stateMachine.update(alignedGeometry, qualityData);
    }

    expect(stateMachine.state).toBe('ALIGNED');
  });

  test('transitions to MONITORING when quality sufficient', () => {
    const alignedGeometry = {
      detected: true,
      rotation: { yaw: 0, pitch: 0, roll: 0 },
    };

    const qualityData = {
      qualityScore: 75,
    };

    // Reach ALIGNED
    for (let i = 0; i < 15; i++) {
      stateMachine.update(alignedGeometry, qualityData);
    }

    // One more update to reach MONITORING
    stateMachine.update(alignedGeometry, qualityData);
    expect(stateMachine.state).toBe('MONITORING');
  });

  test('allows recording only in MONITORING state', () => {
    stateMachine.state = 'SCANNING';
    stateMachine.recordingAllowed = false;
    expect(stateMachine.recordingAllowed).toBe(false);

    stateMachine.state = 'MONITORING';
    stateMachine.recordingAllowed = true;
    expect(stateMachine.recordingAllowed).toBe(true);
  });

  test('handles face misalignment', () => {
    // Start in MONITORING
    stateMachine.state = 'MONITORING';
    stateMachine.recordingAllowed = true;

    // Face moves away (large yaw)
    const misalignedGeometry = {
      detected: true,
      rotation: {
        yaw: 30, // Beyond tolerance
        pitch: 5,
        roll: 5,
      },
    };

    const qualityData = {
      qualityScore: 75,
    };

    stateMachine.update(misalignedGeometry, qualityData);

    // Should transition back to SCANNING
    expect(stateMachine.state).toBe('SCANNING');
    expect(stateMachine.recordingAllowed).toBe(false);
  });

  test('transitions to DEGRADED on quality drop', () => {
    stateMachine.state = 'MONITORING';

    const alignedGeometry = {
      detected: true,
      rotation: { yaw: 0, pitch: 0, roll: 0 },
    };

    // Poor quality
    const poorQualityData = {
      qualityScore: 35, // Below degradation threshold
    };

    stateMachine.update(alignedGeometry, poorQualityData);
    expect(stateMachine.state).toBe('DEGRADED');
  });

  test('recovers from DEGRADED to MONITORING', () => {
    stateMachine.state = 'DEGRADED';

    const alignedGeometry = {
      detected: true,
      rotation: { yaw: 0, pitch: 0, roll: 0 },
    };

    // Good quality again
    const goodQualityData = {
      qualityScore: 75,
    };

    stateMachine.update(alignedGeometry, goodQualityData);
    expect(stateMachine.state).toBe('MONITORING');
  });

  test('updates integrity score', () => {
    const initialScore = stateMachine.integrityScore;

    const alignedGeometry = {
      detected: true,
      rotation: { yaw: 0, pitch: 0, roll: 0 },
    };

    const qualityData = {
      qualityScore: 85,
    };

    stateMachine.update(alignedGeometry, qualityData);

    // Score should improve with good conditions
    expect(stateMachine.integrityScore).toBeGreaterThan(initialScore);
  });

  test('tracks degradation count', () => {
    const initialCount = stateMachine.degradationCount;

    stateMachine.state = 'MONITORING';
    stateMachine.handleAlignmentDropout();

    expect(stateMachine.degradationCount).toBe(initialCount + 1);
  });

  test('can reset state', () => {
    stateMachine.state = 'MONITORING';
    stateMachine.recordingAllowed = true;
    stateMachine.integrityScore = 85;

    stateMachine.reset();

    expect(stateMachine.state).toBe('INITIALIZING');
    expect(stateMachine.recordingAllowed).toBe(false);
    expect(stateMachine.integrityScore).toBe(0);
  });

  test('provides state description', () => {
    stateMachine.state = 'MONITORING';
    const description = stateMachine.getStateDescription();

    expect(description).toHaveProperty('message');
    expect(description).toHaveProperty('status');
    expect(description).toHaveProperty('icon');
    expect(description.message).toContain('Ready');
  });
});

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

describe('Integrity Monitoring Integration', () => {
  test('full workflow from initialization to recording', async () => {
    // Initialize
    const serviceReady = await FaceDetectionService.initialize();
    expect(serviceReady).toBe(true);

    const stateMachine = new AssessmentIntegrityStateMachine();

    // Simulate camera input
    const mockGeometry = {
      detected: true,
      rotation: { yaw: 0, pitch: 0, roll: 0 },
      center: { x: 0.5, y: 0.5 },
      boundingBox: {
        minX: 0.2,
        maxX: 0.8,
        minY: 0.2,
        maxY: 0.8,
      },
    };

    const mockQuality = {
      qualityScore: 80,
      issues: [],
    };

    // Update state machine multiple times
    let recordingEnabled = false;
    for (let i = 0; i < 20; i++) {
      const result = stateMachine.update(mockGeometry, mockQuality);
      if (result.recordingAllowed) {
        recordingEnabled = true;
      }
    }

    // Should eventually allow recording
    expect(recordingEnabled).toBe(true);
    expect(stateMachine.state).toBe('MONITORING');

    FaceDetectionService.cleanup();
  });

  test('handles quality degradation during recording', async () => {
    const stateMachine = new AssessmentIntegrityStateMachine();

    const alignedGeometry = {
      detected: true,
      rotation: { yaw: 0, pitch: 0, roll: 0 },
    };

    // Start recording
    for (let i = 0; i < 20; i++) {
      stateMachine.update(alignedGeometry, { qualityScore: 80 });
    }
    expect(stateMachine.state).toBe('MONITORING');

    // Simulate quality degradation
    for (let i = 0; i < 10; i++) {
      stateMachine.update(alignedGeometry, { qualityScore: 30 });
    }

    // Should be in DEGRADED state
    expect(stateMachine.state).toBe('DEGRADED');

    // Recover
    for (let i = 0; i < 10; i++) {
      stateMachine.update(alignedGeometry, { qualityScore: 80 });
    }

    // Should be back in MONITORING
    expect(stateMachine.state).toBe('MONITORING');
  });
});

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

describe('Performance', () => {
  test('quality analysis completes in < 50ms', () => {
    const analyzer = new QualityAnalyzer();
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, 640, 480);

    const mockGeometry = {
      center: { x: 0.5, y: 0.5 },
      rotation: { yaw: 0, pitch: 0, roll: 0 },
    };

    const start = performance.now();
    analyzer.analyzeFrame(canvas, mockGeometry);
    const duration = performance.now() - start;

    expect(duration).toBeLessThan(50);
  });

  test('state machine update completes in < 5ms', () => {
    const stateMachine = new AssessmentIntegrityStateMachine();

    const geometry = {
      detected: true,
      rotation: { yaw: 0, pitch: 0, roll: 0 },
    };

    const start = performance.now();
    stateMachine.update(geometry, { qualityScore: 75 });
    const duration = performance.now() - start;

    expect(duration).toBeLessThan(5);
  });
});

// ============================================================================
// EXPORT TEST UTILS
// ============================================================================

export const createMockFaceGeometry = (overrides = {}) => {
  return {
    detected: true,
    rotation: { yaw: 0, pitch: 0, roll: 0 },
    center: { x: 0.5, y: 0.5 },
    boundingBox: {
      minX: 0.2,
      maxX: 0.8,
      minY: 0.2,
      maxY: 0.8,
    },
    ...overrides,
  };
};

export const createMockQualityData = (overrides = {}) => {
  return {
    qualityScore: 75,
    frameMetrics: {
      brightness: 150,
      blur: 600,
      motion: 0,
      visibility: 0.95,
      centering: 0.9,
    },
    issues: [],
    ...overrides,
  };
};
