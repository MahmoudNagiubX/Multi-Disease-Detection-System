// ============================================
// MDDS - Professional Medical Platform
// Main JavaScript - Version 4.0
// ============================================

'use strict';

// ============================================
// DARK MODE - Pill Toggle Version
// ============================================
const DarkMode = {
  init() {
    // Load saved theme or detect system preference
    const savedTheme = localStorage.getItem('mdds-theme');
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = savedTheme || (systemPrefersDark ? 'dark' : 'light');
    
    this.setTheme(theme);
    this.setupToggle();
    this.watchSystemPreference();
  },
  
  setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('mdds-theme', theme);
    this.updateToggleUI(theme);
  },
  
  toggle() {
    const current = document.documentElement.getAttribute('data-theme');
    const newTheme = current === 'dark' ? 'light' : 'dark';
    this.setTheme(newTheme);
  },
  
  setupToggle() {
    const toggles = document.querySelectorAll('.pill-toggle');
    toggles.forEach(toggle => {
      toggle.addEventListener('click', () => this.toggle());
    });
  },
  
  updateToggleUI(theme) {
    const toggles = document.querySelectorAll('.pill-toggle');
    toggles.forEach(toggle => {
      const options = toggle.querySelectorAll('.pill-option');
      options.forEach(opt => {
        if (opt.dataset.theme === theme) {
          opt.classList.add('active');
        } else {
          opt.classList.remove('active');
        }
      });
    });
  },
  
  watchSystemPreference() {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
      if (!localStorage.getItem('mdds-theme')) {
        this.setTheme(e.matches ? 'dark' : 'light');
      }
    });
  }
};

// ============================================
// SIDEBAR NAVIGATION
// ============================================
const Sidebar = {
  init() {
    const toggle = document.getElementById('menuToggle');
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.querySelector('.main-content');
    
    if (toggle && sidebar) {
      toggle.addEventListener('click', (e) => {
        e.stopPropagation();
        sidebar.classList.toggle('open');
      });
      
      // Close sidebar when clicking outside on mobile
      document.addEventListener('click', (e) => {
        if (window.innerWidth <= 1024 && 
            sidebar.classList.contains('open') && 
            !sidebar.contains(e.target) && 
            e.target !== toggle) {
          sidebar.classList.remove('open');
        }
      });
    }
  }
};

// ============================================
// FORM ENHANCEMENTS
// ============================================
const Forms = {
  init() {
    this.setupValidation();
    this.setupFileUploads();
  },
  
  setupValidation() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
      form.addEventListener('submit', (e) => {
        const submitBtn = form.querySelector('[type="submit"]');
        if (submitBtn) {
          submitBtn.disabled = true;
          submitBtn.textContent = 'Processing...';
        }
      });
    });
  },
  
  setupFileUploads() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
      input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
          console.log('File selected:', file.name);
        }
      });
    });
  }
};

// ============================================
// NOTIFICATIONS
// ============================================
const Notifications = {
  show(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      padding: 16px 24px;
      background: var(--color-surface);
      border: 1px solid var(--color-border);
      border-radius: 8px;
      box-shadow: var(--shadow-lg);
      z-index: 9999;
      animation: slideIn 0.3s ease-out;
      font-weight: 500;
      color: var(--color-text);
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
      notification.style.opacity = '0';
      notification.style.transform = 'translateY(-10px)';
      notification.style.transition = 'all 0.3s ease-out';
      setTimeout(() => notification.remove(), 300);
    }, 3000);
  }
};

// ============================================
// INITIALIZE ON DOM READY
// ============================================
document.addEventListener('DOMContentLoaded', () => {
  DarkMode.init();
  Sidebar.init();
  Forms.init();
  
  // Convert flash messages to notifications
  document.querySelectorAll('.flash').forEach(flash => {
    const message = flash.textContent.trim();
    const type = flash.classList.contains('flash-success') ? 'success' : 'error';
    Notifications.show(message, type);
    flash.remove();
  });
});
