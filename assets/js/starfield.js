class StarField {
  constructor() {
    this.container = null;
    this.stars = [];
    this.shootingStars = [];
    this.floatingStars = [];
    this.isVisible = true;
    this.shootingStarAngle = 0;
    this.init();
  }

  init() {
    this.createContainer();
    this.calculateShootingStarAngle();

    if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      this.createStars();
      this.createShootingStars();
      this.createFloatingStars();
      this.startAnimations();
    }

    this.addVisibilityListener();
  }

  calculateShootingStarAngle() {
    // Calculate the angle based on screen dimensions
    // Movement: from (100vw, -100vh) to (-100vw, 100vh)
    const deltaX = -200; // -100vw to 100vw = -200vw
    const deltaY = 200;  // -100vh to 100vh = 200vh
    
    // Convert to actual pixels for accurate angle
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    
    const actualDeltaX = deltaX * (vw / 100);
    const actualDeltaY = deltaY * (vh / 100);
    
    // Calculate angle in radians, then convert to degrees
    const angleRad = Math.atan2(actualDeltaY, actualDeltaX);
    this.shootingStarAngle = angleRad * (180 / Math.PI);
    
    // Update CSS custom property for the angle
    document.documentElement.style.setProperty(
      '--shooting-star-angle', 
      `${this.shootingStarAngle}deg`
    );
  }

  createContainer() {
    this.container = document.createElement('div');
    this.container.className = 'starfield';
    this.container.id = 'starfield';

    document.body.insertBefore(this.container, document.body.firstChild);
  }

  createStars() {
    const starCount = window.innerWidth < 768 ? 80 : 150;

    for (let i = 0; i < starCount; i++) {
      const star = document.createElement('div');
      star.className = `star ${this.getRandomSize()}`;
      star.style.left = Math.random() * 100 + '%';
      star.style.top = Math.random() * 100 + '%';
      star.style.animationDelay = Math.random() * 3 + 's';

      this.container.appendChild(star);
      this.stars.push(star);
    }
  }

  createShootingStars() {
    const shootingStarCount = window.innerWidth < 768 ? 2 : 3;

    for (let i = 0; i < shootingStarCount; i++) {
      const shootingStar = document.createElement('div');
      shootingStar.className = 'shooting-star';
      this.randomizeShootingStar(shootingStar);
      shootingStar.style.animationDelay = Math.random() * 8 + 's';
      shootingStar.style.animationDuration = 5 + Math.random() * 4 + 's';

      this.container.appendChild(shootingStar);
      this.shootingStars.push(shootingStar);
    }
  }

  createFloatingStars() {
    const floatingStarCount = window.innerWidth < 768 ? 5 : 10;

    for (let i = 0; i < floatingStarCount; i++) {
      const floatingStar = document.createElement('div');
      floatingStar.className = 'floating-star star small';
      floatingStar.style.left = Math.random() * 100 + '%';
      floatingStar.style.animationDelay = Math.random() * 15 + 's';

      this.container.appendChild(floatingStar);
      this.floatingStars.push(floatingStar);
    }
  }

  randomizeShootingStar(star) {
    star.style.left = Math.random() * 100 + '%';
    star.style.top = Math.random() * 50 + '%';
  }

  getRandomSize() {
    const sizes = ['small', 'medium', 'large'];
    const weights = [0.7, 0.25, 0.05];
    const random = Math.random();
    let cumulative = 0;

    for (let i = 0; i < sizes.length; i++) {
      cumulative += weights[i];
      if (random <= cumulative) return sizes[i];
    }
    return 'small';
  }

  startAnimations() {
    setInterval(() => {
      if (this.isVisible) {
        this.shootingStars.forEach((star) => {
          this.randomizeShootingStar(star);
        });
      }
    }, 8000);

    setInterval(() => {
      if (this.isVisible) {
        this.floatingStars.forEach((star) => {
          star.style.left = Math.random() * 100 + '%';
        });
      }
    }, 15000);
  }

  addVisibilityListener() {
    document.addEventListener('visibilitychange', () => {
      this.isVisible = !document.hidden;
      const playState = this.isVisible ? 'running' : 'paused';

      if (this.container) {
        this.container.style.animationPlayState = playState;
        this.container.querySelectorAll('*').forEach((el) => {
          el.style.animationPlayState = playState;
        });
      }
    });
  }

  destroy() {
    if (this.container && this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
    }
  }

  // Method to recalculate angle on resize
  handleResize() {
    this.calculateShootingStarAngle();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const excludePages = ['print'];
  const shouldInit = !excludePages.some((id) =>
    document.body.classList.contains(id) || document.body.id === id
  );

  if (shouldInit) {
    window.starField = new StarField();
  }
});

window.addEventListener('resize', () => {
  if (window.starField) {
    clearTimeout(window.starField.resizeTimeout);
    window.starField.resizeTimeout = setTimeout(() => {
      // Recalculate angle before destroying and recreating
      window.starField.handleResize();
      window.starField.destroy();
      window.starField = new StarField();
    }, 250);
  }
});