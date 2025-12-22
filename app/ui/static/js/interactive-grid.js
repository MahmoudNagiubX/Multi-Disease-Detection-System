/**
 * Particle Network Background
 * Canvas-based interactive particle network with randomized dots connected by lines
 * Uses GSAP for smooth cursor tracking
 */

class ElasticGrid {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      console.error('Grid container not found:', containerId);
      return;
    }

    // Check if GSAP is available
    if (typeof gsap === 'undefined') {
      console.error('GSAP is required but not loaded. Please include GSAP script.');
      return;
    }

    this.canvas = null;
    this.ctx = null;
    this.animationFrameId = null;
    
    // Particle configuration
    this.PARTICLE_COUNT = 150; // Number of random particles
    this.CONNECTION_DISTANCE = 150; // Max distance to draw connection lines
    this.REPULSION_RADIUS = 180;
    this.REPULSION_STRENGTH = 25;
    this.SPRING_TENSION = 0.03;
    this.FRICTION = 0.92;
    this.PARTICLE_RADIUS = 2.5;

    // Smoothed mouse position (tracked via GSAP)
    this.mouse = { x: -9999, y: -9999 };

    // Particle state
    this.particles = [];

    this.init();
  }

  init() {
    this.createCanvas();
    this.initParticles();
    this.setupEventListeners();
    this.animate();
  }

  createCanvas() {
    // Remove any existing canvas
    const existingCanvas = this.container.querySelector('canvas');
    if (existingCanvas) {
      existingCanvas.remove();
    }

    // Create canvas element
    this.canvas = document.createElement('canvas');
    this.canvas.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    `;
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
  }

  initParticles() {
    const width = window.innerWidth;
    const height = window.innerHeight;

    this.canvas.width = width;
    this.canvas.height = height;

    this.particles = [];

    // Create randomly positioned particles
    for (let i = 0; i < this.PARTICLE_COUNT; i++) {
      const ox = Math.random() * width;
      const oy = Math.random() * height;
      
      this.particles.push({
        x: ox,
        y: oy,
        ox: ox, // origin x
        oy: oy, // origin y
        vx: 0,  // velocity x
        vy: 0   // velocity y
      });
    }
  }

  setupEventListeners() {
    // Mouse tracking with GSAP smoothing
    const handleMouseMove = (e) => {
      gsap.to(this.mouse, {
        x: e.clientX,
        y: e.clientY,
        duration: 0.2,
        ease: 'power2.out'
      });
    };

    const handleResize = () => {
      this.initParticles();
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('resize', handleResize);

    // Store handlers for cleanup
    this._handleMouseMove = handleMouseMove;
    this._handleResize = handleResize;
  }

  animate() {
    const width = this.canvas.width;
    const height = this.canvas.height;
    const { x: mx, y: my } = this.mouse;
    const radiusSq = this.REPULSION_RADIUS * this.REPULSION_RADIUS;

    // Clear canvas
    this.ctx.clearRect(0, 0, width, height);

    // Update physics for each particle
    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];

      // Vector from cursor to particle
      const dx = p.x - mx;
      const dy = p.y - my;
      const distSq = dx * dx + dy * dy;

      // Apply repulsion force if within radius
      if (distSq < radiusSq) {
        const dist = Math.sqrt(distSq) || 0.0001;
        const force = ((this.REPULSION_RADIUS - dist) / this.REPULSION_RADIUS) * this.REPULSION_STRENGTH;

        const nx = dx / dist; // normalized direction
        const ny = dy / dist;

        p.vx += nx * force;
        p.vy += ny * force;
      }

      // Spring back to origin
      const sx = p.ox - p.x;
      const sy = p.oy - p.y;

      p.vx += sx * this.SPRING_TENSION;
      p.vy += sy * this.SPRING_TENSION;

      // Apply friction
      p.vx *= this.FRICTION;
      p.vy *= this.FRICTION;

      // Integrate velocity to position
      p.x += p.vx;
      p.y += p.vy;
    }

    // Draw connection lines between nearby particles
    const connectionDistSq = this.CONNECTION_DISTANCE * this.CONNECTION_DISTANCE;
    
    for (let i = 0; i < this.particles.length; i++) {
      for (let j = i + 1; j < this.particles.length; j++) {
        const p1 = this.particles[i];
        const p2 = this.particles[j];

        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const distSq = dx * dx + dy * dy;

        if (distSq < connectionDistSq) {
          const dist = Math.sqrt(distSq);
          const alpha = 1 - (dist / this.CONNECTION_DISTANCE);
          
          this.ctx.beginPath();
          this.ctx.strokeStyle = `rgba(139, 92, 246, ${alpha * 0.3})`;
          this.ctx.lineWidth = 1;
          this.ctx.moveTo(p1.x, p1.y);
          this.ctx.lineTo(p2.x, p2.y);
          this.ctx.stroke();
        }
      }
    }

    // Draw particles as dots
    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      
      // Check if particle is displaced
      const dx = p.x - p.ox;
      const dy = p.y - p.oy;
      const displacement = Math.sqrt(dx * dx + dy * dy);
      
      // Larger and brighter if displaced
      if (displacement > 5) {
        this.ctx.fillStyle = 'rgba(139, 92, 246, 0.9)';
        this.ctx.shadowBlur = 10;
        this.ctx.shadowColor = 'rgba(139, 92, 246, 0.8)';
      } else {
        this.ctx.fillStyle = 'rgba(139, 92, 246, 0.5)';
        this.ctx.shadowBlur = 0;
      }
      
      this.ctx.beginPath();
      this.ctx.arc(p.x, p.y, this.PARTICLE_RADIUS, 0, Math.PI * 2);
      this.ctx.fill();
      
      // Reset shadow
      this.ctx.shadowBlur = 0;
    }

    // Continue animation loop
    this.animationFrameId = requestAnimationFrame(() => this.animate());
  }

  destroy() {
    // Cleanup event listeners
    if (this._handleMouseMove) {
      window.removeEventListener('mousemove', this._handleMouseMove);
    }
    if (this._handleResize) {
      window.removeEventListener('resize', this._handleResize);
    }

    // Cancel animation frame
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
    }

    // Remove canvas
    if (this.canvas && this.canvas.parentNode) {
      this.canvas.parentNode.removeChild(this.canvas);
    }
  }
}

// Initialize when DOM is ready (or immediately if DOM is already loaded)
function initElasticGridIfPresent() {
  const gridContainer = document.getElementById('interactive-grid-container');
  if (gridContainer && !gridContainer.__gridInitialized) {
    gridContainer.__gridInitialized = true;
    new ElasticGrid('interactive-grid-container');
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initElasticGridIfPresent);
} else {
  // DOMContentLoaded has already fired
  initElasticGridIfPresent();
}
