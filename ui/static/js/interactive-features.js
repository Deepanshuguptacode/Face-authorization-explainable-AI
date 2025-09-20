/**
 * Advanced Interactive Features for Face Recognition Dashboard
 * =========================================================
 * Enhanced interactivity with prototypes, counterfactuals, and sensitivity analysis
 */

class InteractiveFeatureManager {
    constructor() {
        this.currentMode = null;
        this.interactiveData = null;
        this.animationDuration = 300;
        this.isLoading = false;
    }

    /**
     * Initialize interactive features
     */
    initialize(interactiveData) {
        this.interactiveData = interactiveData;
        this.setupEventListeners();
        console.log('Interactive features initialized');
    }

    /**
     * Setup event listeners for interactive buttons
     */
    setupEventListeners() {
        // Prototype analysis
        document.getElementById('showPrototypesBtn')?.addEventListener('click', () => {
            this.showPrototypeAnalysis();
        });

        // Counterfactual modifications
        document.getElementById('addGlassesBtn')?.addEventListener('click', () => {
            this.showCounterfactualAnalysis('glasses');
        });

        document.getElementById('changeExpressionBtn')?.addEventListener('click', () => {
            this.showCounterfactualAnalysis('expression');
        });

        // Advanced features
        document.getElementById('sensitivityAnalysisBtn')?.addEventListener('click', () => {
            this.showSensitivityAnalysis();
        });

        document.getElementById('featureImportanceBtn')?.addEventListener('click', () => {
            this.showFeatureImportance();
        });
    }

    /**
     * Show prototype analysis with enhanced visualization
     */
    async showPrototypeAnalysis() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.setActiveMode('prototypes');
        
        const content = document.getElementById('interactiveContent');
        content.innerHTML = this.createLoadingIndicator('Loading similar prototypes...');

        try {
            // Simulate API call for prototypes
            await this.delay(800);
            
            const prototypes = await this.fetchPrototypeData();
            content.innerHTML = this.renderPrototypeAnalysis(prototypes);
            
            // Add interactive features to prototypes
            this.enhancePrototypeInteractivity();
            
        } catch (error) {
            console.error('Error loading prototypes:', error);
            content.innerHTML = this.createErrorMessage('Failed to load prototype analysis');
        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Fetch prototype data (mock implementation)
     */
    async fetchPrototypeData() {
        return [
            {
                id: 1,
                similarity: 0.894,
                confidence: 0.95,
                attributes: ['Male', 'Young', 'Professional', 'Clean_Shaven'],
                description: 'Young professional male prototype',
                frequency: '12.3% of dataset',
                accuracy: '94.2%',
                keyFeatures: [
                    'Defined jawline structure',
                    'Professional grooming',
                    'Clear skin texture',
                    'Standard lighting conditions'
                ]
            },
            {
                id: 2,
                similarity: 0.867,
                confidence: 0.92,
                attributes: ['Male', 'Smiling', 'Expressive', 'Engaging'],
                description: 'Expressive young male prototype',
                frequency: '8.7% of dataset',
                accuracy: '91.7%',
                keyFeatures: [
                    'Positive facial expression',
                    'Engaged eye contact',
                    'Natural muscle configuration',
                    'Enhanced feature visibility'
                ]
            },
            {
                id: 3,
                similarity: 0.823,
                confidence: 0.89,
                attributes: ['Male', 'Casual', 'Relaxed', 'Natural'],
                description: 'Casual natural male prototype',
                frequency: '6.4% of dataset',
                accuracy: '88.9%',
                keyFeatures: [
                    'Relaxed facial muscles',
                    'Natural pose and angle',
                    'Informal presentation',
                    'Consistent feature alignment'
                ]
            }
        ];
    }

    /**
     * Render prototype analysis with enhanced details
     */
    renderPrototypeAnalysis(prototypes) {
        return `
            <div class="prototype-analysis-container">
                <div class="row mb-4">
                    <div class="col-12">
                        <h5 class="mb-3">
                            <i class="fas fa-users me-2 text-primary"></i>
                            Similar Face Prototypes
                        </h5>
                        <p class="text-muted">
                            These prototypes represent common face patterns that match your input. 
                            Click on any prototype for detailed analysis.
                        </p>
                    </div>
                </div>
                
                <div class="row">
                    ${prototypes.map(proto => this.renderPrototypeCard(proto)).join('')}
                </div>
                
                <div class="prototype-summary mt-4">
                    <div class="card">
                        <div class="card-header bg-light">
                            <h6 class="mb-0">
                                <i class="fas fa-chart-bar me-2"></i>
                                Prototype Analysis Summary
                            </h6>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-4">
                                    <div class="text-center">
                                        <div class="h4 text-primary">${prototypes.length}</div>
                                        <div class="text-muted">Matching Prototypes</div>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="text-center">
                                        <div class="h4 text-success">${(prototypes[0].similarity * 100).toFixed(1)}%</div>
                                        <div class="text-muted">Best Match</div>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="text-center">
                                        <div class="h4 text-info">${prototypes[0].accuracy}</div>
                                        <div class="text-muted">Accuracy Rate</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render individual prototype card
     */
    renderPrototypeCard(prototype) {
        return `
            <div class="col-lg-4 mb-4">
                <div class="prototype-card h-100" data-prototype-id="${prototype.id}">
                    <div class="prototype-header">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h6 class="mb-0">Prototype #${prototype.id}</h6>
                            <span class="badge bg-primary">${(prototype.similarity * 100).toFixed(1)}%</span>
                        </div>
                        
                        <!-- Prototype Image Placeholder -->
                        <div class="prototype-image-container mb-3">
                            <div class="prototype-placeholder">
                                <i class="fas fa-user fa-3x text-muted"></i>
                                <div class="mt-2 small text-muted">Prototype Face</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="prototype-details">
                        <div class="mb-3">
                            <strong>Description:</strong>
                            <div class="text-muted small">${prototype.description}</div>
                        </div>
                        
                        <div class="mb-3">
                            <strong>Key Attributes:</strong>
                            <div class="attribute-tags mt-1">
                                ${prototype.attributes.map(attr => 
                                    `<span class="badge bg-light text-dark me-1 mb-1">${attr}</span>`
                                ).join('')}
                            </div>
                        </div>
                        
                        <div class="prototype-stats">
                            <div class="row small">
                                <div class="col-6">
                                    <div><strong>Dataset Frequency:</strong></div>
                                    <div class="text-muted">${prototype.frequency}</div>
                                </div>
                                <div class="col-6">
                                    <div><strong>Accuracy:</strong></div>
                                    <div class="text-muted">${prototype.accuracy}</div>
                                </div>
                            </div>
                        </div>
                        
                        <button class="btn btn-outline-primary btn-sm mt-3 w-100" 
                                onclick="interactiveManager.showPrototypeDetails(${prototype.id})">
                            <i class="fas fa-info-circle me-1"></i>
                            View Details
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Show detailed prototype information
     */
    showPrototypeDetails(prototypeId) {
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-user me-2"></i>
                            Prototype #${prototypeId} Analysis
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        ${this.renderPrototypeDetailContent(prototypeId)}
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
        
        // Clean up modal after hiding
        modal.addEventListener('hidden.bs.modal', () => {
            document.body.removeChild(modal);
        });
    }

    /**
     * Render detailed prototype content
     */
    renderPrototypeDetailContent(prototypeId) {
        const details = {
            1: {
                title: 'Young Professional Male',
                technicalDescription: 'This prototype represents the most common pattern for young professional males in our dataset.',
                keyFeatures: [
                    'Clean facial structure with defined jawline',
                    'Professional grooming standards',
                    'Clear skin indicative of younger demographic',
                    'Standard professional lighting conditions',
                    'Frontal pose with direct eye contact'
                ],
                statisticalData: {
                    'Dataset Coverage': '12.3% of training data',
                    'Recognition Accuracy': '94.2% average',
                    'False Positive Rate': '2.1%',
                    'Confidence Threshold': '0.85',
                    'Feature Stability': '91.7%'
                },
                variations: [
                    'With/without eyeglasses (±3.2% accuracy)',
                    'Different hair colors (±1.8% accuracy)',
                    'Various lighting conditions (±5.1% accuracy)',
                    'Minor expression changes (±2.4% accuracy)'
                ]
            }
        };
        
        const detail = details[prototypeId] || details[1];
        
        return `
            <div class="prototype-detail-content">
                <div class="row">
                    <div class="col-md-4">
                        <div class="prototype-image-large">
                            <div class="placeholder-large">
                                <i class="fas fa-user fa-4x text-muted"></i>
                                <div class="mt-3">Prototype Face</div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-8">
                        <h6>Technical Description</h6>
                        <p class="text-muted">${detail.technicalDescription}</p>
                        
                        <h6>Key Features</h6>
                        <ul class="list-unstyled">
                            ${detail.keyFeatures.map(feature => 
                                `<li><i class="fas fa-check text-success me-2"></i>${feature}</li>`
                            ).join('')}
                        </ul>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-6">
                        <h6>Statistical Data</h6>
                        <table class="table table-sm">
                            ${Object.entries(detail.statisticalData).map(([key, value]) => 
                                `<tr><td><strong>${key}:</strong></td><td>${value}</td></tr>`
                            ).join('')}
                        </table>
                    </div>
                    <div class="col-md-6">
                        <h6>Common Variations</h6>
                        <ul class="list-unstyled">
                            ${detail.variations.map(variation => 
                                `<li><i class="fas fa-arrow-right text-primary me-2"></i>${variation}</li>`
                            ).join('')}
                        </ul>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Show counterfactual analysis
     */
    async showCounterfactualAnalysis(modificationType) {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.setActiveMode('counterfactual');
        
        const content = document.getElementById('interactiveContent');
        content.innerHTML = this.createLoadingIndicator(`Analyzing ${modificationType} modification...`);

        try {
            await this.delay(600);
            
            const analysis = await this.generateCounterfactualData(modificationType);
            content.innerHTML = this.renderCounterfactualAnalysis(analysis);
            
            // Add interactive charts
            this.createSensitivityChart(analysis.sensitivityData);
            
        } catch (error) {
            console.error('Error in counterfactual analysis:', error);
            content.innerHTML = this.createErrorMessage('Failed to generate counterfactual analysis');
        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Generate counterfactual data
     */
    async generateCounterfactualData(modificationType) {
        const baseScore = 0.752;
        
        const modifications = {
            glasses: {
                name: 'Eyeglasses',
                originalScore: baseScore,
                modifiedScore: baseScore - 0.054,
                impact: -0.054,
                confidence: 0.91,
                explanation: 'Adding glasses creates occlusion around the critical eye region, reducing recognition accuracy.',
                technicalDetails: 'The periocular region contributes approximately 35% to face recognition performance.',
                affectedRegions: ['eye_region', 'nose_bridge', 'upper_face']
            },
            expression: {
                name: 'Facial Expression',
                originalScore: baseScore,
                modifiedScore: baseScore + 0.037,
                impact: +0.037,
                confidence: 0.88,
                explanation: 'Adding a smile improves feature visibility and creates more distinctive facial configurations.',
                technicalDetails: 'Smiling activates 12-15 facial muscles, creating additional geometric features.',
                affectedRegions: ['mouth_region', 'cheek_region', 'eye_region']
            }
        };
        
        const data = modifications[modificationType] || modifications.glasses;
        
        // Generate sensitivity data
        data.sensitivityData = this.generateSensitivityData(modificationType);
        
        return data;
    }

    /**
     * Generate sensitivity analysis data
     */
    generateSensitivityData(modificationType) {
        const baseModifications = [
            { name: 'Add Glasses', impact: -0.054, confidence: 0.91 },
            { name: 'Add Smile', impact: +0.037, confidence: 0.88 },
            { name: 'Change Lighting', impact: -0.029, confidence: 0.83 },
            { name: 'Add Beard', impact: -0.023, confidence: 0.85 },
            { name: 'Change Pose', impact: -0.032, confidence: 0.91 },
            { name: 'Add Makeup', impact: -0.015, confidence: 0.82 }
        ];
        
        return baseModifications;
    }

    /**
     * Render counterfactual analysis
     */
    renderCounterfactualAnalysis(analysis) {
        const isImprovement = analysis.impact > 0;
        const impactClass = isImprovement ? 'text-success' : 'text-danger';
        const impactIcon = isImprovement ? 'arrow-up' : 'arrow-down';
        
        return `
            <div class="counterfactual-analysis-container">
                <div class="row mb-4">
                    <div class="col-12">
                        <h5 class="mb-3">
                            <i class="fas fa-exchange-alt me-2 text-primary"></i>
                            Counterfactual Analysis: ${analysis.name}
                        </h5>
                        <p class="text-muted">${analysis.explanation}</p>
                    </div>
                </div>
                
                <!-- Before/After Comparison -->
                <div class="comparison-container mb-4">
                    <div class="row">
                        <div class="col-md-5">
                            <div class="comparison-card original">
                                <div class="card">
                                    <div class="card-header bg-light">
                                        <h6 class="mb-0">Original Prediction</h6>
                                    </div>
                                    <div class="card-body text-center">
                                        <div class="score-display">
                                            <div class="score-value">${analysis.originalScore.toFixed(3)}</div>
                                            <div class="score-label">Similarity Score</div>
                                        </div>
                                        <div class="confidence-indicator mt-3">
                                            <small class="text-muted">
                                                Confidence: ${(analysis.confidence * 100).toFixed(1)}%
                                            </small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="col-md-2 d-flex align-items-center justify-content-center">
                            <div class="comparison-arrow">
                                <i class="fas fa-arrow-right fa-2x text-muted"></i>
                            </div>
                        </div>
                        
                        <div class="col-md-5">
                            <div class="comparison-card modified">
                                <div class="card">
                                    <div class="card-header ${isImprovement ? 'bg-success' : 'bg-warning'} text-white">
                                        <h6 class="mb-0">With ${analysis.name}</h6>
                                    </div>
                                    <div class="card-body text-center">
                                        <div class="score-display">
                                            <div class="score-value">${analysis.modifiedScore.toFixed(3)}</div>
                                            <div class="score-label">Similarity Score</div>
                                        </div>
                                        <div class="impact-indicator mt-3">
                                            <span class="impact-change ${impactClass}">
                                                <i class="fas fa-${impactIcon} me-1"></i>
                                                ${analysis.impact > 0 ? '+' : ''}${analysis.impact.toFixed(3)}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Technical Analysis -->
                <div class="technical-analysis mb-4">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">
                                <i class="fas fa-cogs me-2"></i>
                                Technical Analysis
                            </h6>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>Impact Assessment</h6>
                                    <p class="text-muted">${analysis.technicalDetails}</p>
                                    
                                    <div class="impact-metrics">
                                        <div class="metric-item">
                                            <span class="metric-label">Magnitude:</span>
                                            <span class="metric-value">
                                                ${Math.abs(analysis.impact) > 0.05 ? 'Large' : 
                                                  Math.abs(analysis.impact) > 0.02 ? 'Medium' : 'Small'}
                                            </span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-label">Direction:</span>
                                            <span class="metric-value ${impactClass}">
                                                ${isImprovement ? 'Improvement' : 'Degradation'}
                                            </span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-label">Confidence:</span>
                                            <span class="metric-value">${(analysis.confidence * 100).toFixed(1)}%</span>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <h6>Sensitivity Analysis</h6>
                                    <canvas id="sensitivityChart" width="400" height="250"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Comprehensive Sensitivity Table -->
                <div class="sensitivity-table">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">
                                <i class="fas fa-table me-2"></i>
                                Model Sensitivity to Various Modifications
                            </h6>
                        </div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-hover">
                                    <thead>
                                        <tr>
                                            <th>Modification</th>
                                            <th>Impact</th>
                                            <th>Confidence</th>
                                            <th>Magnitude</th>
                                            <th>Effect</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${analysis.sensitivityData.map(item => `
                                            <tr>
                                                <td>${item.name}</td>
                                                <td class="${item.impact > 0 ? 'text-success' : 'text-danger'}">
                                                    ${item.impact > 0 ? '+' : ''}${item.impact.toFixed(3)}
                                                </td>
                                                <td>${(item.confidence * 100).toFixed(1)}%</td>
                                                <td>
                                                    <div class="progress" style="height: 8px;">
                                                        <div class="progress-bar ${item.impact > 0 ? 'bg-success' : 'bg-danger'}" 
                                                             style="width: ${Math.abs(item.impact) * 1000}%"></div>
                                                    </div>
                                                </td>
                                                <td>
                                                    <span class="badge ${item.impact > 0 ? 'bg-success' : 'bg-danger'}">
                                                        ${item.impact > 0 ? 'Improves' : 'Reduces'}
                                                    </span>
                                                </td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Create sensitivity chart
     */
    createSensitivityChart(sensitivityData) {
        setTimeout(() => {
            const canvas = document.getElementById('sensitivityChart');
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Chart parameters
            const margin = { top: 20, right: 20, bottom: 60, left: 80 };
            const chartWidth = canvas.width - margin.left - margin.right;
            const chartHeight = canvas.height - margin.top - margin.bottom;
            
            // Prepare data
            const sortedData = sensitivityData.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
            const maxImpact = Math.max(...sortedData.map(d => Math.abs(d.impact)));
            
            // Draw bars
            const barHeight = chartHeight / sortedData.length * 0.8;
            const barSpacing = chartHeight / sortedData.length * 0.2;
            
            sortedData.forEach((item, index) => {
                const y = margin.top + index * (barHeight + barSpacing);
                const barWidth = (Math.abs(item.impact) / maxImpact) * chartWidth;
                const x = item.impact > 0 ? margin.left : margin.left - barWidth;
                
                // Bar color
                ctx.fillStyle = item.impact > 0 ? '#28a745' : '#dc3545';
                ctx.fillRect(margin.left, y, barWidth, barHeight);
                
                // Label
                ctx.fillStyle = '#495057';
                ctx.font = '12px sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText(item.name, margin.left - 5, y + barHeight / 2 + 4);
                
                // Value
                ctx.textAlign = 'left';
                ctx.fillText(
                    `${item.impact > 0 ? '+' : ''}${item.impact.toFixed(3)}`,
                    margin.left + barWidth + 5,
                    y + barHeight / 2 + 4
                );
            });
            
            // Center line
            ctx.strokeStyle = '#6c757d';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(margin.left, margin.top);
            ctx.lineTo(margin.left, margin.top + chartHeight);
            ctx.stroke();
            
            // Title
            ctx.fillStyle = '#495057';
            ctx.font = 'bold 14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Impact on Similarity Score', canvas.width / 2, 15);
            
        }, 100);
    }

    /**
     * Utility methods
     */
    setActiveMode(mode) {
        this.currentMode = mode;
        
        // Update button states
        document.querySelectorAll('.interactive-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        const activeBtn = document.querySelector(`[data-mode="${mode}"]`);
        if (activeBtn) {
            activeBtn.classList.add('active');
        }
    }

    createLoadingIndicator(message = 'Loading...') {
        return `
            <div class="loading-container text-center py-5">
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="text-muted">${message}</p>
            </div>
        `;
    }

    createErrorMessage(message) {
        return `
            <div class="error-container text-center py-5">
                <div class="text-danger mb-3">
                    <i class="fas fa-exclamation-triangle fa-3x"></i>
                </div>
                <p class="text-muted">${message}</p>
                <button class="btn btn-primary" onclick="location.reload()">
                    <i class="fas fa-refresh me-2"></i>Retry
                </button>
            </div>
        `;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Enhance prototype interactivity
     */
    enhancePrototypeInteractivity() {
        // Add hover effects
        document.querySelectorAll('.prototype-card').forEach(card => {
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-5px)';
                card.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
            });
            
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0)';
                card.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
            });
        });
    }
}

// Initialize interactive feature manager
const interactiveManager = new InteractiveFeatureManager();

// Export for global access
window.interactiveManager = interactiveManager;