/ static/js/main.js

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('Football AI App Loaded');
    
    // Initialize components
    initNavigation();
    initFilters();
    initCharts();
    initAnimations();
    initLiveUpdates();
});

// Navigation Handler
function initNavigation() {
    const navToggle = document.getElementById('navToggle');
    const navMenu = document.querySelector('.nav-menu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', () => {
            navMenu.classList.toggle('active');
            navToggle.classList.toggle('active');
        });
    }
    
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
    
    // Active page highlighting
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
}

// Filter System
function initFilters() {
    // League filter
    const leagueFilter = document.getElementById('leagueFilter');
    if (leagueFilter) {
        leagueFilter.addEventListener('change', applyFilters);
    }
    
    // Confidence filter
    const confidenceFilter = document.getElementById('confidenceFilter');
    if (confidenceFilter) {
        confidenceFilter.addEventListener('change', applyFilters);
    }
    
    // Date filter
    const dateFilter = document.getElementById('dateFilter');
    if (dateFilter) {
        dateFilter.addEventListener('change', applyFilters);
    }
    
    // Period filter
    const periodFilter = document.getElementById('periodFilter');
    if (periodFilter) {
        periodFilter.addEventListener('change', applyFilters);
    }
}

function applyFilters() {
    const params = new URLSearchParams();
    
    // Collect all filter values
    const filters = {
        league: document.getElementById('leagueFilter')?.value,
        confidence: document.getElementById('confidenceFilter')?.value,
        date: document.getElementById('dateFilter')?.value,
        period: document.getElementById('periodFilter')?.value,
        correct: document.getElementById('correctOnly')?.checked
    };
    
    // Add non-empty filters to params
    Object.keys(filters).forEach(key => {
        if (filters[key] && filters[key] !== '') {
            params.set(key, filters[key]);
        }
    });
    
    // Update URL without reload
    const newUrl = window.location.pathname + '?' + params.toString();
    window.history.pushState({}, '', newUrl);
    
    // Apply filters to cards (client-side filtering)
    filterCards(filters);
}

function filterCards(filters) {
    const cards = document.querySelectorAll('.prediction-card, .result-card');
    
    cards.forEach(card => {
        let show = true;
        
        // Check league filter
        if (filters.league && card.dataset.league !== filters.league) {
            show = false;
        }
        
        // Check confidence filter
        if (filters.confidence === 'high' && card.dataset.confidence !== '1') {
            show = false;
        }
        
        // Check date filter
        if (filters.date && card.dataset.date !== filters.date) {
            show = false;
        }
        
        // Show/hide card with animation
        if (show) {
            card.style.display = 'block';
            card.classList.add('fade-in');
        } else {
            card.style.display = 'none';
        }
    });
    
    // Update results count
    updateResultsCount();
}

function updateResultsCount() {
    const visibleCards = document.querySelectorAll('.prediction-card:not([style*="display: none"]), .result-card:not([style*="display: none"])');
    const countElement = document.getElementById('resultsCount');
    
    if (countElement) {
        countElement.textContent = `${visibleCards.length} Ergebnisse`;
    }
}

// Charts Initialization
function initCharts() {
    // Check if we're on the dashboard
    if (!document.getElementById('accuracyChart')) {
        return;
    }
    
    // Fetch statistics data
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            drawAccuracyChart(data);
            drawROIChart(data);
        })
        .catch(error => console.error('Error loading stats:', error));
}

function drawAccuracyChart(data) {
    const canvas = document.getElementById('accuracyChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Simple bar chart
    const values = [
        data.all_time.accuracy,
        data.last_month.accuracy,
        data.last_week.accuracy
    ];
    
    const labels = ['Gesamt', 'Monat', 'Woche'];
    const barWidth = width / values.length * 0.6;
    const spacing = width / values.length;
    
    ctx.fillStyle = '#3b82f6';
    values.forEach((value, index) => {
        const barHeight = (value / 100) * height * 0.8;
        const x = index * spacing + (spacing - barWidth) / 2;
        const y = height - barHeight;
        
        ctx.fillRect(x, y, barWidth, barHeight);
        
        // Draw label
        ctx.fillStyle = '#1f2937';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(labels[index], x + barWidth / 2, height - 5);
        ctx.fillText(`${value}%`, x + barWidth / 2, y - 5);
        
        ctx.fillStyle = '#3b82f6';
    });
}

function drawROIChart(data) {
    const canvas = document.getElementById('roiChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Line chart for ROI
    const values = [
        data.all_time.roi,
        data.last_month.roi,
        data.last_week.roi
    ];
    
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    values.forEach((value, index) => {
        const x = (index / (values.length - 1)) * width;
        const y = height - ((value + 50) / 100 * height); // Adjust for negative values
        
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
        
        // Draw point
        ctx.fillStyle = '#10b981';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
    });
    
    ctx.stroke();
}

// Animations
function initAnimations() {
    // Intersection Observer for fade-in animations
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1
    });
    
    // Observe all cards
    document.querySelectorAll('.card, .stat-card, .prediction-card, .result-card').forEach(card => {
        observer.observe(card);
    });
    
    // Counter animations for statistics
    animateCounters();
}

function animateCounters() {
    const counters = document.querySelectorAll('[data-counter]');
    
    counters.forEach(counter => {
        const target = parseInt(counter.dataset.counter);
        const duration = 1000; // 1 second
        const steps = 50;
        const increment = target / steps;
        let current = 0;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                counter.textContent = target;
                clearInterval(timer);
            } else {
                counter.textContent = Math.floor(current);
            }
        }, duration / steps);
    });
}

// Live Updates
function initLiveUpdates() {
    // Update statistics every 30 seconds
    setInterval(updateStatistics, 30000);
    
    // Check for new predictions every minute
    setInterval(checkNewPredictions, 60000);
}

function updateStatistics() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            // Update dashboard statistics
            updateStatElement('total-bets', data.all_time.total_bets);
            updateStatElement('accuracy', data.all_time.accuracy + '%');
            updateStatElement('roi', data.all_time.roi + '%');
        })
        .catch(error => console.error('Error updating stats:', error));
}

function updateStatElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
        element.classList.add('fade-in');
    }
}

function checkNewPredictions() {
    fetch('/api/predictions/latest')
        .then(response => response.json())
        .then(data => {
            const countElement = document.getElementById('prediction-count');
            if (countElement && data.count) {
                const currentCount = parseInt(countElement.textContent);
                if (data.count > currentCount) {
                    showNotification('Neue Vorhersagen verfügbar!');
                    countElement.textContent = data.count;
                }
            }
        })
        .catch(error => console.error('Error checking predictions:', error));
}

// Notifications
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type} slide-in`;
    notification.textContent = message;
    
    // Style the notification
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        z-index: 9999;
        min-width: 250px;
    `;
    
    document.body.appendChild(notification);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Utility Functions
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('de-DE', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric'
    });
}

function formatTime(timeString) {
    return timeString.substring(0, 5);
}

function calculatePercentage(value, total) {
    if (total === 0) return 0;
    return Math.round((value / total) * 100);
}

// Export functions for global use
window.FootballAI = {
    showNotification,
    formatDate,
    formatTime,
    calculatePercentage,
    applyFilters,
    updateStatistics
};