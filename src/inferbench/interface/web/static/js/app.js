// InferBench Framework JavaScript

// API helper functions
const api = {
    async get(endpoint) {
        const response = await fetch(endpoint);
        return response.json();
    },
    
    async post(endpoint, data) {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        return response.json();
    },
    
    async delete(endpoint) {
        const response = await fetch(endpoint, {method: 'DELETE'});
        return response.json();
    }
};

// Status badge helper
function getStatusBadge(status) {
    const classes = {
        'running': 'success',
        'completed': 'success',
        'pending': 'warning',
        'queued': 'warning',
        'error': 'danger',
        'failed': 'danger',
        'stopped': 'secondary',
        'canceled': 'secondary'
    };
    const badgeClass = classes[status] || 'secondary';
    return `<span class="badge bg-${badgeClass}">${status}</span>`;
}

// Time formatting
function formatTime(isoString) {
    if (!isoString) return '-';
    return new Date(isoString).toLocaleString();
}

// Short ID
function shortId(id) {
    return id ? id.substring(0, 8) + '...' : '-';
}

// Toast notifications
function showToast(message, type = 'info') {
    // Simple alert for now, can be enhanced with Bootstrap toasts
    alert(message);
}

// Console logging with prefix
function log(...args) {
    console.log('[InferBench]', ...args);
}

// Initialize tooltips
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Bootstrap tooltips if present
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipTriggerList.forEach(el => new bootstrap.Tooltip(el));
    
    log('Application initialized');
});
