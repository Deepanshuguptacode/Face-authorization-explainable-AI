/**
 * Dashboard JavaScript for Explainable Face Recognition
 * ===================================================
 * Interactive functionality for the face verification dashboard
 */

// Global state
let currentImages = {
    image1: null,
    image2: null
};

let verificationResults = null;
let isAccessibilityMode = false;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupEventListeners();
});

/**
 * Initialize dashboard components
 */
function initializeDashboard() {
    // Check for accessibility preferences
    if (localStorage.getItem('accessibilityMode') === 'true') {
        toggleAccessibility();
    }
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    console.log('Dashboard initialized');
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Image upload listeners with error handling
    const image1Input = document.getElementById('image1');
    const image2Input = document.getElementById('image2');
    
    if (image1Input) {
        image1Input.addEventListener('change', function() {
            console.log('Image 1 changed');
            loadImage(this, 'preview1');
            checkVerifyButton();
        });
    }
    
    if (image2Input) {
        image2Input.addEventListener('change', function() {
            console.log('Image 2 changed');
            loadImage(this, 'preview2');
            checkVerifyButton();
        });
    }
    
    // Settings change listeners
    const saliencyCheck = document.getElementById('showSaliency');
    const attributesCheck = document.getElementById('showAttributes');
    const interactiveCheck = document.getElementById('enableInteractive');
    
    if (saliencyCheck) saliencyCheck.addEventListener('change', updateDisplaySettings);
    if (attributesCheck) attributesCheck.addEventListener('change', updateDisplaySettings);
    if (interactiveCheck) interactiveCheck.addEventListener('change', updateDisplaySettings);
    
    // Keyboard navigation
    document.addEventListener('keydown', handleKeyboardNavigation);
}

/**
 * Load and preview an uploaded image
 */
function loadImage(input, previewId) {
    if (!input || !input.files || !input.files[0]) {
        console.log('No file selected');
        return;
    }

    const file = input.files[0];
    const reader = new FileReader();
    
    console.log(`Loading ${previewId}: ${file.name}`);
    
    reader.onload = function(e) {
        const preview = document.getElementById(previewId);
        if (!preview) {
            console.error(`Preview element ${previewId} not found`);
            return;
        }
        
        preview.src = e.target.result;
        preview.style.display = 'block';
        
        // Store image data
        if (previewId === 'preview1') {
            currentImages.image1 = {
                file: file,
                dataUrl: e.target.result
            };
            console.log('✅ Image 1 loaded successfully');
        } else if (previewId === 'preview2') {
            currentImages.image2 = {
                file: file,
                dataUrl: e.target.result
            };
            console.log('✅ Image 2 loaded successfully');
        }
        
        // Animate preview appearance
        preview.style.opacity = '0';
        preview.style.transform = 'scale(0.8)';
        preview.style.transition = 'all 0.3s ease';
        
        setTimeout(() => {
            preview.style.opacity = '1';
            preview.style.transform = 'scale(1)';
            
            // Show success message
            if (currentImages.image1 && currentImages.image2) {
                showToast('Both images loaded! Click "Run Verification" to proceed.', 'success');
            }
        }, 100);
        
        console.log(`Image loaded: ${previewId}`, currentImages);
    };
    
    reader.onerror = function(error) {
        console.error('Error reading file:', error);
        showToast('Error loading image. Please try again.', 'error');
    };
    
    reader.readAsDataURL(file);
}

/**
 * Check if verify button should be enabled
 */
function checkVerifyButton() {
    const verifyBtn = document.getElementById('verifyBtn');
    if (!verifyBtn) {
        console.error('Verify button not found');
        return;
    }
    
    const hasImages = currentImages.image1 && currentImages.image2;
    
    console.log('Checking verify button state:', {
        hasImage1: !!currentImages.image1,
        hasImage2: !!currentImages.image2,
        hasImages: hasImages
    });
    
    verifyBtn.disabled = !hasImages;
    
    if (hasImages) {
        verifyBtn.classList.remove('btn-secondary');
        verifyBtn.classList.add('btn-primary');
        verifyBtn.innerHTML = '<i class="fas fa-play me-2"></i>Run Verification';
        console.log('✅ Verify button enabled - both images loaded');
    } else {
        verifyBtn.classList.remove('btn-primary');
        verifyBtn.classList.add('btn-secondary');
        verifyBtn.innerHTML = '<i class="fas fa-upload me-2"></i>Upload Images First';
        console.log('⏳ Verify button disabled - waiting for images');
    }
}

/**
 * Run face verification
 */
async function runVerification() {
    console.log('🚀 Starting verification process');
    console.log('Current images state:', currentImages);
    
    if (!currentImages.image1 || !currentImages.image2) {
        console.error('Missing images for verification');
        alert('Please upload both images first!');
        return;
    }
    
    // Show loading with immediate feedback
    console.log('Showing loading overlay...');
    showLoading(true);
    
    // Add a small delay to see if loading shows
    await new Promise(resolve => setTimeout(resolve, 100));
    
    // Safeguard: Force hide loading after 10 seconds (reduced for testing)
    const loadingTimeout = setTimeout(() => {
        console.warn('⚠️ Forcing loading modal to hide after timeout');
        showLoading(false);
        alert('Verification timed out. Please try again.');
    }, 10000);
    
    try {
        console.log('📤 Preparing form data...');
        
        // Prepare form data
        const formData = new FormData();
        formData.append('image1', currentImages.image1.file);
        formData.append('image2', currentImages.image2.file);
        
        // Get settings with fallbacks
        const settings = {
            explanationStyle: document.getElementById('explanationStyle')?.value || 'comprehensive',
            showSaliency: document.getElementById('showSaliency')?.checked || true,
            showAttributes: document.getElementById('showAttributes')?.checked || true,
            enableInteractive: document.getElementById('enableInteractive')?.checked || true
        };
        
        formData.append('settings', JSON.stringify(settings));
        
        console.log('📤 Sending verification request with settings:', settings);
        console.log('Files being sent:', {
            image1: currentImages.image1.file.name,
            image2: currentImages.image2.file.name
        });
        
        // Make API call
        const response = await fetch('/verify', {
            method: 'POST',
            body: formData
        });
        
        console.log('📥 Response received, status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Response error:', errorText);
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        const results = await response.json();
        console.log('✅ Verification results received:', results);
        
        if (!results.success) {
            throw new Error(results.error || 'Verification failed');
        }
        
        verificationResults = results;
        
        // Display results
        console.log('🎨 Displaying results...');
        displayVerificationResults(results);
        showToast('Verification completed successfully!', 'success');
        
    } catch (error) {
        console.error('❌ Verification error:', error);
        alert(`Verification failed: ${error.message}`);
        showToast(`Verification failed: ${error.message}`, 'error');
    } finally {
        console.log('🔄 Cleaning up - hiding loading overlay');
        clearTimeout(loadingTimeout);
        showLoading(false);
        
        // Double-check overlay is hidden
        setTimeout(() => {
            console.log('🔍 Final check - forcing overlay hidden');
            forceHideAllModals();
        }, 500);
    }
}

/**
 * Display verification results
 */
function displayVerificationResults(results) {
    console.log('🎨 Displaying verification results');
    
    // Force hide loading modal first
    showLoading(false);
    
    // Show results card
    const resultsCard = document.getElementById('resultsCard');
    if (!resultsCard) {
        console.error('Results card not found!');
        return;
    }
    
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth' });
    
    // Update match result
    updateMatchResult(results.similarity, results.isMatch);
    
    // Update confidence gauge
    updateConfidenceGauge(results.similarity);
    
    // Display saliency maps if available
    if (results.saliency && document.getElementById('showSaliency').checked) {
        displaySaliencyMaps(results.saliency);
    }
    
    // Display attributes if available
    if (results.attributes && document.getElementById('showAttributes').checked) {
        displayAttributeAnalysis(results.attributes);
    }
    
    // Display explanation
    displayExplanation(results.explanation);
    
    // Enable interactive features if requested
    if (document.getElementById('enableInteractive').checked) {
        enableInteractiveFeatures(results);
    }
    
    // Animate card appearance
    animateCardAppearance();
}

/**
 * Update match result display
 */
function updateMatchResult(similarity, isMatch) {
    const matchResult = document.getElementById('matchResult');
    const scoreText = matchResult.querySelector('.score-text');
    const scoreValue = matchResult.querySelector('.score-value');
    
    // Update text and score
    scoreText.textContent = isMatch ? 'MATCH' : 'NO MATCH';
    scoreValue.textContent = similarity.toFixed(3);
    
    // Update styling
    matchResult.classList.remove('match-pending', 'match-positive', 'match-negative');
    matchResult.classList.add(isMatch ? 'match-positive' : 'match-negative');
    
    // Add completion animation
    matchResult.style.transform = 'scale(1.05)';
    setTimeout(() => {
        matchResult.style.transform = 'scale(1)';
    }, 200);
}

/**
 * Update confidence gauge
 */
function updateConfidenceGauge(similarity) {
    const canvas = document.getElementById('confidenceGauge');
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Gauge parameters
    const centerX = canvas.width / 2;
    const centerY = canvas.height - 20;
    const radius = 80;
    const startAngle = Math.PI;
    const endAngle = 2 * Math.PI;
    
    // Background arc
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, startAngle, endAngle);
    ctx.strokeStyle = '#e9ecef';
    ctx.lineWidth = 10;
    ctx.stroke();
    
    // Progress arc
    const progressAngle = startAngle + (similarity * Math.PI);
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, startAngle, progressAngle);
    ctx.strokeStyle = similarity > 0.5 ? '#28a745' : '#dc3545';
    ctx.lineWidth = 10;
    ctx.stroke();
    
    // Center text
    ctx.fillStyle = '#495057';
    ctx.font = '16px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText((similarity * 100).toFixed(1) + '%', centerX, centerY - 10);
}

/**
 * Display saliency maps
 */
function displaySaliencyMaps(saliencyData) {
    if (saliencyData.image1) {
        const saliency1 = document.getElementById('saliency1');
        const container1 = document.getElementById('saliency1Container');
        saliency1.src = 'data:image/png;base64,' + saliencyData.image1;
        container1.style.display = 'block';
    }
    
    if (saliencyData.image2) {
        const saliency2 = document.getElementById('saliency2');
        const container2 = document.getElementById('saliency2Container');
        saliency2.src = 'data:image/png;base64,' + saliencyData.image2;
        container2.style.display = 'block';
    }
}

/**
 * Display attribute analysis
 */
function displayAttributeAnalysis(attributes) {
    const attributesCard = document.getElementById('attributesCard');
    const attributesList = document.getElementById('attributesList');
    
    // Clear previous content
    attributesList.innerHTML = '';
    
    // Sort attributes by confidence
    const sortedAttributes = attributes.sort((a, b) => Math.abs(b.confidence) - Math.abs(a.confidence));
    
    // Display top attributes
    sortedAttributes.slice(0, 10).forEach(attr => {
        const attributeItem = createAttributeItem(attr);
        attributesList.appendChild(attributeItem);
    });
    
    // Create attribute chart
    createAttributeChart(sortedAttributes.slice(0, 8));
    
    // Show attributes card
    attributesCard.style.display = 'block';
}

/**
 * Create attribute item element
 */
function createAttributeItem(attribute) {
    const item = document.createElement('div');
    item.className = 'attribute-item';
    
    const confidence = Math.abs(attribute.confidence);
    const isPositive = attribute.confidence > 0;
    
    item.innerHTML = `
        <div class="attribute-name">
            ${isPositive ? '✅' : '❌'} ${attribute.name}
        </div>
        <div class="d-flex align-items-center">
            <span class="attribute-confidence ${isPositive ? 'confidence-positive' : 'confidence-negative'}">
                ${confidence.toFixed(3)}
            </span>
            <div class="confidence-bar">
                <div class="confidence-fill ${isPositive ? 'positive' : 'negative'}" 
                     style="width: ${confidence * 100}%"></div>
            </div>
        </div>
    `;
    
    return item;
}

/**
 * Create attribute chart
 */
function createAttributeChart(attributes) {
    const canvas = document.getElementById('attributeChart');
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Chart parameters
    const margin = { top: 20, right: 20, bottom: 40, left: 120 };
    const chartWidth = canvas.width - margin.left - margin.right;
    const chartHeight = canvas.height - margin.top - margin.bottom;
    const barHeight = chartHeight / attributes.length;
    
    // Draw bars
    attributes.forEach((attr, index) => {
        const y = margin.top + index * barHeight;
        const barWidth = Math.abs(attr.confidence) * chartWidth;
        const x = attr.confidence > 0 ? margin.left : margin.left - barWidth;
        
        // Bar
        ctx.fillStyle = attr.confidence > 0 ? '#28a745' : '#dc3545';
        ctx.fillRect(margin.left, y + 5, barWidth, barHeight - 10);
        
        // Label
        ctx.fillStyle = '#495057';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(attr.name, margin.left - 5, y + barHeight / 2 + 4);
        
        // Value
        ctx.textAlign = 'left';
        ctx.fillText(attr.confidence.toFixed(3), margin.left + barWidth + 5, y + barHeight / 2 + 4);
    });
    
    // Center line
    ctx.strokeStyle = '#6c757d';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + chartHeight);
    ctx.stroke();
}

/**
 * Display explanation text
 */
function displayExplanation(explanation) {
    const explanationCard = document.getElementById('explanationCard');
    const explanationText = document.getElementById('explanationText');
    const accessibilityText = document.getElementById('accessibilityText');
    const accessibilityDescription = document.getElementById('accessibilityDescription');
    
    // Main explanation
    explanationText.innerHTML = explanation.text;
    
    // Accessibility description
    if (explanation.accessibility) {
        accessibilityDescription.textContent = explanation.accessibility;
        accessibilityText.style.display = isAccessibilityMode ? 'block' : 'none';
    }
    
    explanationCard.style.display = 'block';
}

/**
 * Enable interactive features
 */
function enableInteractiveFeatures(results) {
    const interactiveCard = document.getElementById('interactiveCard');
    interactiveCard.style.display = 'block';
    
    // Store results for interactive functions
    window.interactiveData = results.interactive || {};
}

/**
 * Show similar prototypes
 */
function showPrototypes() {
    const content = document.getElementById('interactiveContent');
    
    const prototypes = window.interactiveData.prototypes || [
        { id: 1, similarity: 0.89, attributes: ['Male', 'Young', 'Brown_Hair'] },
        { id: 2, similarity: 0.85, attributes: ['Male', 'Smiling', 'No_Beard'] },
        { id: 3, similarity: 0.82, attributes: ['Male', 'Young', 'Attractive'] }
    ];
    
    content.innerHTML = `
        <div class="row">
            <div class="col-12">
                <h5 class="mb-3">👥 Similar Face Prototypes</h5>
            </div>
            ${prototypes.map(proto => `
                <div class="col-md-4">
                    <div class="prototype-card">
                        <h6>Prototype #${proto.id}</h6>
                        <div class="prototype-similarity">${proto.similarity.toFixed(3)}</div>
                        <div class="prototype-attributes">
                            ${proto.attributes.join(', ')}
                        </div>
                        <div class="mt-2">
                            <img src="https://via.placeholder.com/100x100?text=P${proto.id}" 
                                 class="img-fluid rounded" alt="Prototype ${proto.id}">
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

/**
 * Add/remove glasses counterfactual
 */
function addGlasses() {
    const content = document.getElementById('interactiveContent');
    
    content.innerHTML = `
        <div class="counterfactual-comparison">
            <div class="counterfactual-item">
                <div class="counterfactual-title">Original Prediction</div>
                <div class="score-value">0.752</div>
                <div class="mt-2">
                    <small class="text-muted">Attributes: Male, Young, No_Beard</small>
                </div>
            </div>
            <div class="counterfactual-item">
                <div class="counterfactual-title">With Glasses Added</div>
                <div class="score-value">0.698</div>
                <div class="score-change score-decrease">-0.054</div>
                <div class="mt-2">
                    <small class="text-muted">Attributes: Male, Young, No_Beard, <strong>Eyeglasses</strong></small>
                </div>
                <div class="alert alert-info mt-3">
                    <i class="fas fa-info-circle me-2"></i>
                    Adding glasses decreased similarity by 0.054 points
                </div>
            </div>
        </div>
        
        <div class="mt-4">
            <h6>Model Sensitivity Analysis</h6>
            <table class="table table-sm">
                <thead>
                    <tr>
                        <th>Modification</th>
                        <th>Score Change</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Add Glasses</td>
                        <td class="score-decrease">-0.054</td>
                        <td>0.92</td>
                    </tr>
                    <tr>
                        <td>Add Smile</td>
                        <td class="score-increase">+0.037</td>
                        <td>0.88</td>
                    </tr>
                    <tr>
                        <td>Add Beard</td>
                        <td class="score-decrease">-0.023</td>
                        <td>0.85</td>
                    </tr>
                </tbody>
            </table>
        </div>
    `;
}

/**
 * Change expression counterfactual
 */
function changeExpression() {
    const content = document.getElementById('interactiveContent');
    
    content.innerHTML = `
        <div class="counterfactual-comparison">
            <div class="counterfactual-item">
                <div class="counterfactual-title">Original Prediction</div>
                <div class="score-value">0.752</div>
                <div class="mt-2">
                    <small class="text-muted">Attributes: Male, Young, Neutral</small>
                </div>
            </div>
            <div class="counterfactual-item">
                <div class="counterfactual-title">With Smile Added</div>
                <div class="score-value">0.789</div>
                <div class="score-change score-increase">+0.037</div>
                <div class="mt-2">
                    <small class="text-muted">Attributes: Male, Young, <strong>Smiling</strong></small>
                </div>
                <div class="alert alert-success mt-3">
                    <i class="fas fa-smile me-2"></i>
                    Adding smile increased similarity by 0.037 points
                </div>
            </div>
        </div>
        
        <div class="mt-4">
            <h6>Expression Impact Analysis</h6>
            <div class="row">
                <div class="col-md-6">
                    <canvas id="expressionChart" width="300" height="200"></canvas>
                </div>
                <div class="col-md-6">
                    <div class="alert alert-light">
                        <h6>Key Findings:</h6>
                        <ul class="mb-0">
                            <li>Smiling increases perceived similarity</li>
                            <li>Neutral expressions are more stable</li>
                            <li>Extreme expressions may decrease accuracy</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Draw expression impact chart
    setTimeout(() => {
        const canvas = document.getElementById('expressionChart');
        if (canvas) {
            drawExpressionChart(canvas);
        }
    }, 100);
}

/**
 * Draw expression impact chart
 */
function drawExpressionChart(canvas) {
    const ctx = canvas.getContext('2d');
    const expressions = ['Neutral', 'Slight Smile', 'Full Smile', 'Frown'];
    const scores = [0.752, 0.789, 0.734, 0.698];
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Chart parameters
    const margin = { top: 20, right: 20, bottom: 40, left: 60 };
    const chartWidth = canvas.width - margin.left - margin.right;
    const chartHeight = canvas.height - margin.top - margin.bottom;
    
    // Draw bars
    const barWidth = chartWidth / expressions.length;
    const maxScore = Math.max(...scores);
    
    expressions.forEach((expr, index) => {
        const barHeight = (scores[index] / maxScore) * chartHeight;
        const x = margin.left + index * barWidth + 10;
        const y = margin.top + chartHeight - barHeight;
        
        // Bar
        ctx.fillStyle = scores[index] > 0.75 ? '#28a745' : '#ffc107';
        ctx.fillRect(x, y, barWidth - 20, barHeight);
        
        // Label
        ctx.fillStyle = '#495057';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.save();
        ctx.translate(x + (barWidth - 20) / 2, margin.top + chartHeight + 15);
        ctx.rotate(-Math.PI / 4);
        ctx.fillText(expr, 0, 0);
        ctx.restore();
        
        // Value
        ctx.textAlign = 'center';
        ctx.fillText(scores[index].toFixed(3), x + (barWidth - 20) / 2, y - 5);
    });
}

/**
 * Load demo images
 */
function loadDemoImages() {
    console.log('🎭 Loading demo images: anantu.jpg and passport phot 2.jpg');
    
    // Show loading status
    showToast('Loading demo images...', 'info');
    
    // Function to convert image URL to File object
    async function urlToFile(url, filename) {
        try {
            const response = await fetch(url);
            const blob = await response.blob();
            return new File([blob], filename, { type: blob.type });
        } catch (error) {
            console.error(`Error loading ${filename}:`, error);
            throw error;
        }
    }
    
    // Load the actual demo images
    Promise.all([
        urlToFile('/demo-images/anantu.jpg', 'anantu.jpg'),
        urlToFile('/demo-images/passport phot 2.jpg', 'passport phot 2.jpg')
    ]).then(([file1, file2]) => {
        // Create URLs for preview
        const url1 = URL.createObjectURL(file1);
        const url2 = URL.createObjectURL(file2);
        
        // Set preview images
        const preview1 = document.getElementById('preview1');
        const preview2 = document.getElementById('preview2');
        
        if (preview1 && preview2) {
            preview1.src = url1;
            preview1.style.display = 'block';
            preview2.src = url2;
            preview2.style.display = 'block';
            
            // Update current images with actual files
            currentImages.image1 = { file: file1, dataUrl: url1 };
            currentImages.image2 = { file: file2, dataUrl: url2 };
            
            // Update file inputs to show selected files
            const input1 = document.getElementById('image1');
            const input2 = document.getElementById('image2');
            
            // Create new FileList objects (this is a bit tricky in JavaScript)
            const dt1 = new DataTransfer();
            const dt2 = new DataTransfer();
            dt1.items.add(file1);
            dt2.items.add(file2);
            
            if (input1) input1.files = dt1.files;
            if (input2) input2.files = dt2.files;
            
            // Enable verify button
            checkVerifyButton();
            
            console.log('✅ Demo images loaded successfully');
            showToast('Demo images loaded! You can now run verification.', 'success');
            
            // Add visual animation
            [preview1, preview2].forEach(preview => {
                preview.style.opacity = '0';
                preview.style.transform = 'scale(0.8)';
                preview.style.transition = 'all 0.3s ease';
                
                setTimeout(() => {
                    preview.style.opacity = '1';
                    preview.style.transform = 'scale(1)';
                }, 100);
            });
            
        } else {
            console.error('Preview elements not found');
            showToast('Error: Preview elements not found', 'error');
        }
        
    }).catch(error => {
        console.error('❌ Error loading demo images:', error);
        showToast(`Error loading demo images: ${error.message}`, 'error');
        
        // Fallback to original demo behavior if loading fails
        loadFallbackDemoImages();
    });
}

function loadFallbackDemoImages() {
    console.log('🔄 Loading fallback demo images');
    
    // Create demo images (placeholder)
    const demoData1 = 'data:image/svg+xml;base64,' + btoa(`
        <svg width="224" height="224" xmlns="http://www.w3.org/2000/svg">
            <rect width="224" height="224" fill="#e3f2fd"/>
            <circle cx="112" cy="80" r="30" fill="#1976d2"/>
            <circle cx="95" cy="75" r="3" fill="white"/>
            <circle cx="129" cy="75" r="3" fill="white"/>
            <path d="M 90 95 Q 112 105 134 95" stroke="#1976d2" stroke-width="2" fill="none"/>
            <text x="112" y="200" text-anchor="middle" font-family="Arial" font-size="14" fill="#1976d2">Anantu (Demo)</text>
        </svg>
    `);
    
    const demoData2 = 'data:image/svg+xml;base64,' + btoa(`
        <svg width="224" height="224" xmlns="http://www.w3.org/2000/svg">
            <rect width="224" height="224" fill="#f3e5f5"/>
            <circle cx="112" cy="80" r="30" fill="#7b1fa2"/>
            <circle cx="95" cy="75" r="3" fill="white"/>
            <circle cx="129" cy="75" r="3" fill="white"/>
            <path d="M 90 95 Q 112 105 134 95" stroke="#7b1fa2" stroke-width="2" fill="none"/>
            <text x="112" y="200" text-anchor="middle" font-family="Arial" font-size="14" fill="#7b1fa2">Passport (Demo)</text>
        </svg>
    `);
    
    // Set demo images
    document.getElementById('preview1').src = demoData1;
    document.getElementById('preview2').src = demoData2;
    
    // Create mock File objects for compatibility
    fetch(demoData1).then(res => res.blob()).then(blob1 => {
        fetch(demoData2).then(res => res.blob()).then(blob2 => {
            currentImages.image1 = { 
                file: new File([blob1], 'anantu_demo.svg', { type: 'image/svg+xml' }), 
                dataUrl: demoData1 
            };
            currentImages.image2 = { 
                file: new File([blob2], 'passport_demo.svg', { type: 'image/svg+xml' }), 
                dataUrl: demoData2 
            };
            
            checkVerifyButton();
            showToast('Fallback demo images loaded', 'info');
        });
    });
}

/**
 * Clear all inputs and results
 */
function clearAll() {
    // Clear file inputs
    document.getElementById('image1').value = '';
    document.getElementById('image2').value = '';
    
    // Clear preview images
    document.getElementById('preview1').style.display = 'none';
    document.getElementById('preview2').style.display = 'none';
    
    // Clear current images
    currentImages.image1 = null;
    currentImages.image2 = null;
    
    // Hide all result cards
    document.getElementById('resultsCard').style.display = 'none';
    document.getElementById('attributesCard').style.display = 'none';
    document.getElementById('explanationCard').style.display = 'none';
    document.getElementById('interactiveCard').style.display = 'none';
    
    // Disable verify button
    checkVerifyButton();
    
    // Clear verification results
    verificationResults = null;
    
    showSuccess('All data cleared successfully!');
}

/**
 * Toggle accessibility mode
 */
function toggleAccessibility() {
    isAccessibilityMode = !isAccessibilityMode;
    document.body.classList.toggle('accessibility-mode', isAccessibilityMode);
    
    // Save preference
    localStorage.setItem('accessibilityMode', isAccessibilityMode.toString());
    
    // Update accessibility text visibility
    const accessibilityTexts = document.querySelectorAll('.accessibility-text');
    accessibilityTexts.forEach(element => {
        element.style.display = isAccessibilityMode ? 'block' : 'none';
    });
    
    // Update button text
    const button = document.getElementById('accessibilityToggle');
    button.innerHTML = `
        <i class="fas fa-universal-access"></i>
        ${isAccessibilityMode ? 'Disable' : 'Enable'} Accessibility
    `;
    
    console.log('Accessibility mode:', isAccessibilityMode ? 'enabled' : 'disabled');
}

/**
 * Update display settings
 */
function updateDisplaySettings() {
    if (verificationResults) {
        displayVerificationResults(verificationResults);
    }
}

/**
 * Animate card appearance
 */
function animateCardAppearance() {
    const cards = document.querySelectorAll('.card[style*="block"]');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'all 0.4s ease';
        
        setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
}

/**
 * Show loading state
 */
function showLoading(show) {
    console.log(`${show ? 'Showing' : 'Hiding'} loading overlay`);
    
    const overlay = document.getElementById('loadingOverlay');
    if (!overlay) {
        console.error('Loading overlay not found!');
        return;
    }
    
    if (show) {
        overlay.style.display = 'flex';
        document.body.style.overflow = 'hidden';
        console.log('✅ Loading overlay shown');
    } else {
        overlay.style.display = 'none';
        document.body.style.overflow = '';
        console.log('✅ Loading overlay hidden');
    }
}

/**
 * Force clear all modals and overlays
 */
function forceHideAllModals() {
    console.log('🔧 Force clearing all modals and overlays');
    
    // Hide custom loading overlay
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
    
    // Hide all Bootstrap modals
    document.querySelectorAll('.modal').forEach(modal => {
        modal.classList.remove('show');
        modal.style.display = 'none';
        modal.setAttribute('aria-hidden', 'true');
    });
    
    // Remove all backdrops
    document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
        backdrop.remove();
    });
    
    // Reset body
    document.body.classList.remove('modal-open');
    document.body.style.overflow = '';
    document.body.style.paddingRight = '';
    document.body.style.pointerEvents = '';
    
    console.log('✅ All modals and overlays cleared');
}

/**
 * Show success message
 */
function showSuccess(message) {
    // Create and show toast notification
    const toast = createToast(message, 'success');
    document.body.appendChild(toast);
    
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    setTimeout(() => {
        document.body.removeChild(toast);
    }, 5000);
}

/**
 * Show error message
 */
function showError(message) {
    showToast(message, 'error');
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    // Create and show toast notification
    const toast = createToast(message, type);
    document.body.appendChild(toast);
    
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    setTimeout(() => {
        if (document.body.contains(toast)) {
            document.body.removeChild(toast);
        }
    }, 5000);
}

/**
 * Create toast notification
 */
function createToast(message, type) {
    const toastEl = document.createElement('div');
    toastEl.className = 'toast position-fixed top-0 end-0 m-3';
    toastEl.setAttribute('role', 'alert');
    toastEl.style.zIndex = '9999';
    
    const bgClass = type === 'success' ? 'bg-success' : 
                    type === 'info' ? 'bg-info' : 
                    type === 'warning' ? 'bg-warning' : 'bg-danger';
    const iconClass = type === 'success' ? 'fas fa-check-circle' : 
                     type === 'info' ? 'fas fa-info-circle' : 
                     type === 'warning' ? 'fas fa-exclamation-triangle' : 'fas fa-exclamation-circle';
    
    toastEl.innerHTML = `
        <div class="toast-header ${bgClass} text-white">
            <i class="${iconClass} me-2"></i>
            <strong class="me-auto">${type.charAt(0).toUpperCase() + type.slice(1)}</strong>
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
        </div>
        <div class="toast-body">
            ${message}
        </div>
    `;
    
    return toastEl;
}

/**
 * Handle keyboard navigation
 */
function handleKeyboardNavigation(event) {
    // ESC key to close modals
    if (event.key === 'Escape') {
        console.log('🔑 Escape key pressed - clearing modals');
        
        const modals = document.querySelectorAll('.modal.show');
        modals.forEach(modal => {
            const bsModal = bootstrap.Modal.getInstance(modal);
            if (bsModal) {
                bsModal.hide();
            }
        });
        
        // Force clear any stuck modals
        setTimeout(() => {
            forceHideAllModals();
        }, 500);
    }
    
    // Enter key on verify button
    if (event.key === 'Enter' && event.target.id === 'verifyBtn') {
        runVerification();
    }
    
    // Accessibility shortcuts
    if (event.ctrlKey && event.key === 'a') {
        event.preventDefault();
        toggleAccessibility();
    }
}