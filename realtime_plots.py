import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RealTimeVisualizer:
    """Real-time visualization of neuro-motor analysis"""
    
    def __init__(self, update_interval=100):  # ms
        self.update_interval = update_interval
        self.fig = None
        self.axes = None
        self.animation = None
        self.data_buffer = []
        self.max_buffer_size = 100
        
        # Color schemes
        self.colors = {
            'normal': '#2E86AB',
            'mild_stress': '#F6AE2D',
            'severe_stress': '#A23B72',
            'tremor': '#FF6B6B',
            'pressure': '#4ECDC4',
            'velocity': '#45B7D1',
            'background': '#F7F9FC'
        }
        
    def setup_plots(self, figsize=(16, 10)):
        """Setup the visualization dashboard"""
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig = plt.figure(figsize=figsize, facecolor=self.colors['background'])
        gs = gridspec.GridSpec(3, 4, figure=self.fig, hspace=0.4, wspace=0.3)
        
        self.axes = {
            'stroke': plt.subplot(gs[0, 0:2]),
            'pressure': plt.subplot(gs[0, 2:]),
            'velocity': plt.subplot(gs[1, 0:2]),
            'tremor': plt.subplot(gs[1, 2:]),
            'npi': plt.subplot(gs[2, 0]),
            'features': plt.subplot(gs[2, 1]),
            'classification': plt.subplot(gs[2, 2]),
            'status': plt.subplot(gs[2, 3])
        }
        
        # Configure each subplot
        self._configure_axes()
        
        plt.suptitle('Neuro-Motor Health Assessment - Real-Time Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        return self.fig, self.axes
    
    def _configure_axes(self):
        """Configure each axis with titles and labels"""
        # Stroke plot
        self.axes['stroke'].set_title('Handwriting Stroke', fontweight='bold')
        self.axes['stroke'].set_xlabel('X Position')
        self.axes['stroke'].set_ylabel('Y Position')
        self.axes['stroke'].grid(True, alpha=0.3)
        self.axes['stroke'].set_aspect('equal')
        
        # Pressure plot
        self.axes['pressure'].set_title('Pressure Analysis', fontweight='bold')
        self.axes['pressure'].set_xlabel('Time (s)')
        self.axes['pressure'].set_ylabel('Pressure')
        self.axes['pressure'].grid(True, alpha=0.3)
        
        # Velocity plot
        self.axes['velocity'].set_title('Velocity Profile', fontweight='bold')
        self.axes['velocity'].set_xlabel('Time (s)')
        self.axes['velocity'].set_ylabel('Velocity')
        self.axes['velocity'].grid(True, alpha=0.3)
        
        # Tremor plot
        self.axes['tremor'].set_title('Tremor Analysis', fontweight='bold')
        self.axes['tremor'].set_xlabel('Frequency (Hz)')
        self.axes['tremor'].set_ylabel('Power')
        self.axes['tremor'].grid(True, alpha=0.3)
        
        # NPI plot
        self.axes['npi'].set_title('NPI Score', fontweight='bold')
        self.axes['npi'].set_ylim(0, 100)
        self.axes['npi'].axhspan(0, 30, alpha=0.3, color='green', label='Normal')
        self.axes['npi'].axhspan(30, 60, alpha=0.3, color='yellow', label='Mild Stress')
        self.axes['npi'].axhspan(60, 100, alpha=0.3, color='red', label='Severe Stress')
        
        # Features plot
        self.axes['features'].set_title('Feature Importance', fontweight='bold')
        
        # Classification plot
        self.axes['classification'].set_title('Classification', fontweight='bold')
        
        # Status plot
        self.axes['status'].set_title('System Status', fontweight='bold')
        self.axes['status'].axis('off')
    
    def update(self, analysis_data):
        """Update plots with new analysis data"""
        self.data_buffer.append(analysis_data)
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer.pop(0)
        
        # Clear axes
        for ax in self.axes.values():
            ax.clear()
        
        # Reconfigure axes
        self._configure_axes()
        
        # Update each plot
        if analysis_data:
            self._update_stroke_plot(analysis_data)
            self._update_pressure_plot(analysis_data)
            self._update_velocity_plot(analysis_data)
            self._update_tremor_plot(analysis_data)
            self._update_npi_plot(analysis_data)
            self._update_features_plot(analysis_data)
            self._update_classification_plot(analysis_data)
            self._update_status_plot(analysis_data)
        
        self.fig.canvas.draw_idle()
    
    def _update_stroke_plot(self, data):
        """Update stroke visualization"""
        if 'strokes' in data:
            for stroke in data['strokes']:
                if len(stroke) >= 2:
                    x = [p.get('x', 0) for p in stroke]
                    y = [p.get('y', 0) for p in stroke]
                    
                    # Color by pressure if available
                    if 'pressure' in stroke[0]:
                        pressures = [p.get('pressure', 0.5) for p in stroke]
                        points = np.array([x, y]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        
                        from matplotlib.collections import LineCollection
                        norm = plt.Normalize(0, 1)
                        lc = LineCollection(segments, cmap='viridis', norm=norm)
                        lc.set_array(np.array(pressures))
                        lc.set_linewidth(2)
                        self.axes['stroke'].add_collection(lc)
                    else:
                        self.axes['stroke'].plot(x, y, 'b-', linewidth=2, alpha=0.7)
            
            self.axes['stroke'].set_xlim(-1.2, 1.2)
            self.axes['stroke'].set_ylim(-1.2, 1.2)
    
    def _update_pressure_plot(self, data):
        """Update pressure analysis plot"""
        if 'pressure_features' in data:
            features = data['pressure_features']
            
            time_points = np.arange(len(self.data_buffer))
            pressure_values = [d.get('pressure_features', {}).get('pressure_mean', 0.5) 
                             for d in self.data_buffer]
            
            self.axes['pressure'].plot(time_points, pressure_values, 
                                      color=self.colors['pressure'], 
                                      linewidth=2, marker='o', markersize=3)
            
            # Add threshold lines
            self.axes['pressure'].axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Light')
            self.axes['pressure'].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Heavy')
            
            self.axes['pressure'].legend()
            self.axes['pressure'].set_xlim(0, self.max_buffer_size)
            self.axes['pressure'].set_ylim(0, 1)
    
    def _update_velocity_plot(self, data):
        """Update velocity profile plot"""
        if 'velocity_features' in data:
            features = data['velocity_features']
            
            time_points = np.arange(len(self.data_buffer))
            velocity_values = [d.get('velocity_features', {}).get('velocity_mean', 0.2) 
                             for d in self.data_buffer]
            velocity_std = [d.get('velocity_features', {}).get('velocity_std', 0.1) 
                          for d in self.data_buffer]
            
            self.axes['velocity'].plot(time_points, velocity_values, 
                                      color=self.colors['velocity'], 
                                      linewidth=2, label='Mean Velocity')
            
            # Add std shading
            self.axes['velocity'].fill_between(time_points,
                                              np.array(velocity_values) - np.array(velocity_std),
                                              np.array(velocity_values) + np.array(velocity_std),
                                              alpha=0.3, color=self.colors['velocity'])
            
            self.axes['velocity'].legend()
            self.axes['velocity'].set_xlim(0, self.max_buffer_size)
            self.axes['velocity'].set_ylim(0, 1)
    
    def _update_tremor_plot(self, data):
        """Update tremor analysis plot"""
        if 'tremor_features' in data:
            features = data['tremor_features']
            
            # Show frequency spectrum if available
            if 'tremor_frequencies' in features and 'tremor_powers' in features:
                freqs = features['tremor_frequencies']
                powers = features['tremor_powers']
                
                self.axes['tremor'].plot(freqs, powers, 
                                        color=self.colors['tremor'], 
                                        linewidth=2)
                
                # Highlight physiological tremor range (3-12 Hz)
                self.axes['tremor'].axvspan(3, 8, alpha=0.2, color='green', label='Normal')
                self.axes['tremor'].axvspan(8, 12, alpha=0.2, color='yellow', label='Elevated')
                self.axes['tremor'].axvspan(12, 20, alpha=0.2, color='red', label='Pathological')
            
            self.axes['tremor'].legend()
            self.axes['tremor'].set_xlim(0, 20)
    
    def _update_npi_plot(self, data):
        """Update NPI score plot"""
        if 'npi_score' in data:
            npi_scores = [d.get('npi_score', 0) for d in self.data_buffer]
            time_points = np.arange(len(npi_scores))
            
            self.axes['npi'].plot(time_points, npi_scores, 
                                 color='black', linewidth=3, marker='o', markersize=4)
            
            # Add current value annotation
            current_npi = npi_scores[-1] if npi_scores else 0
            self.axes['npi'].annotate(f'NPI: {current_npi:.1f}', 
                                     xy=(0.5, 0.9), xycoords='axes fraction',
                                     ha='center', fontsize=12, fontweight='bold',
                                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            self.axes['npi'].set_xlim(0, self.max_buffer_size)
            self.axes['npi'].set_ylim(0, 100)
            self.axes['npi'].set_xlabel('Time')
            self.axes['npi'].set_ylabel('NPI Score')
            self.axes['npi'].legend()
    
    def _update_features_plot(self, data):
        """Update feature importance plot"""
        if 'feature_importance' in data:
            features = data['feature_importance']
            
            if isinstance(features, dict):
                # Get top features
                sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:8]
                feature_names = [f[0] for f in sorted_features]
                importance_values = [f[1] for f in sorted_features]
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(feature_names)))
                self.axes['features'].barh(feature_names, importance_values, color=colors)
                self.axes['features'].set_xlabel('Importance')
            
            self.axes['features'].set_xlim(0, 1)
    
    def _update_classification_plot(self, data):
        """Update classification results plot"""
        if 'classification' in data:
            classification = data['classification']
            
            if isinstance(classification, dict) and 'probabilities' in classification:
                probs = classification['probabilities']
                labels = list(probs.keys())
                values = list(probs.values())
                
                colors = ['green', 'orange', 'red'][:len(labels)]
                wedges, texts, autotexts = self.axes['classification'].pie(
                    values, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=90
                )
                
                # Add center text
                center_text = classification.get('predicted_class', 'Unknown')
                self.axes['classification'].text(0, 0, center_text, 
                                                ha='center', va='center',
                                                fontsize=14, fontweight='bold')
            
            else:
                # Simple text display
                self.axes['classification'].text(0.5, 0.5, 
                                                str(classification),
                                                ha='center', va='center',
                                                fontsize=12)
                self.axes['classification'].axis('off')
    
    def _update_status_plot(self, data):
        """Update system status display"""
        status_text = []
        
        # Timestamp
        timestamp = datetime.now().strftime('%H:%M:%S')
        status_text.append(f"Time: {timestamp}")
        
        # Data quality
        if 'data_quality' in data:
            quality = data['data_quality']
            status_text.append(f"Data Quality: {quality}/100")
        
        # Processing status
        status_text.append("Processing: Active")
        
        # Feature count
        if 'feature_count' in data:
            status_text.append(f"Features: {data['feature_count']}")
        
        # System warnings
        if 'warnings' in data and data['warnings']:
            status_text.append("Warnings: Yes")
        else:
            status_text.append("Warnings: None")
        
        # Display as text
        status_str = "\n".join(status_text)
        self.axes['status'].text(0.1, 0.5, status_str, 
                                fontfamily='monospace', fontsize=10,
                                verticalalignment='center')
        
        # Add colored status indicator
        status_color = 'green'
        if 'npi_score' in data and data['npi_score'] > 60:
            status_color = 'red'
        elif 'npi_score' in data and data['npi_score'] > 30:
            status_color = 'yellow'
        
        self.axes['status'].add_patch(plt.Circle((0.05, 0.9), 0.03, 
                                                color=status_color))
    
    def start_animation(self, data_generator):
        """Start real-time animation"""
        def animate(frame):
            try:
                data = next(data_generator)
                self.update(data)
            except StopIteration:
                pass
            return []
        
        self.animation = FuncAnimation(self.fig, animate, 
                                      interval=self.update_interval,
                                      blit=False, cache_frame_data=False)
        
        plt.show()
    
    def save_dashboard(self, filename='neuro_motor_dashboard.png'):
        """Save current dashboard state"""
        if self.fig:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {filename}")
    
    def create_summary_plot(self, analysis_history):
        """Create summary plot from analysis history"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. NPI trend over time
        npi_scores = [a.get('npi_score', 0) for a in analysis_history]
        time_points = range(len(npi_scores))
        
        axes[0, 0].plot(time_points, npi_scores, 'b-', linewidth=2)
        axes[0, 0].fill_between(time_points, 0, npi_scores, alpha=0.3)
        axes[0, 0].set_title('NPI Score Trend')
        axes[0, 0].set_xlabel('Assessment')
        axes[0, 0].set_ylabel('NPI Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add thresholds
        axes[0, 0].axhline(y=30, color='green', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(y=60, color='orange', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(y=80, color='red', linestyle='--', alpha=0.5)
        
        # 2. Feature correlation heatmap
        if analysis_history and 'features' in analysis_history[0]:
            # Extract common features
            all_features = set()
            for analysis in analysis_history:
                if 'features' in analysis:
                    all_features.update(analysis['features'].keys())
            
            feature_matrix = []
            for analysis in analysis_history:
                if 'features' in analysis:
                    row = [analysis['features'].get(f, 0) for f in all_features]
                    feature_matrix.append(row)
            
            if feature_matrix:
                import seaborn as sns
                corr_matrix = np.corrcoef(np.array(feature_matrix).T)
                
                # Select top correlated features
                top_indices = np.argsort(np.mean(np.abs(corr_matrix), axis=0))[-8:]
                top_features = [list(all_features)[i] for i in top_indices]
                top_corr = corr_matrix[np.ix_(top_indices, top_indices)]
                
                sns.heatmap(top_corr, annot=True, fmt='.2f', cmap='coolwarm',
                           xticklabels=top_features, yticklabels=top_features,
                           ax=axes[0, 1])
                axes[0, 1].set_title('Feature Correlation')
        
        # 3. Stress level distribution
        stress_levels = []
        for analysis in analysis_history:
            if 'classification' in analysis:
                if isinstance(analysis['classification'], dict):
                    level = analysis['classification'].get('predicted_class', 'unknown')
                else:
                    level = analysis['classification']
                stress_levels.append(level)
        
        if stress_levels:
            from collections import Counter
            level_counts = Counter(stress_levels)
            
            labels = list(level_counts.keys())
            counts = list(level_counts.values())
            
            colors = {'normal': 'green', 'mild_stress': 'orange', 
                     'severe_stress': 'red', 'unknown': 'gray'}
            bar_colors = [colors.get(l, 'gray') for l in labels]
            
            axes[0, 2].bar(labels, counts, color=bar_colors)
            axes[0, 2].set_title('Stress Level Distribution')
            axes[0, 2].set_ylabel('Count')
        
        # 4. Feature importance over time
        if analysis_history and 'feature_importance' in analysis_history[0]:
            # Track top features over time
            top_features_timeline = {}
            
            for i, analysis in enumerate(analysis_history):
                if 'feature_importance' in analysis:
                    fi = analysis['feature_importance']
                    if isinstance(fi, dict):
                        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]
                        for feature, importance in sorted_fi:
                            if feature not in top_features_timeline:
                                top_features_timeline[feature] = []
                            top_features_timeline[feature].append((i, importance))
            
            for feature, timeline in top_features_timeline.items():
                if timeline:
                    times, importances = zip(*timeline)
                    axes[1, 0].plot(times, importances, 'o-', label=feature, markersize=4)
            
            axes[1, 0].set_title('Top Feature Importance Over Time')
            axes[1, 0].set_xlabel('Assessment')
            axes[1, 0].set_ylabel('Importance')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Pattern detection
        pattern_counts = {}
        for analysis in analysis_history:
            if 'patterns' in analysis:
                for pattern in analysis['patterns']:
                    if pattern not in pattern_counts:
                        pattern_counts[pattern] = 0
                    pattern_counts[pattern] += 1
        
        if pattern_counts:
            sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:6]
            pattern_names = [p[0] for p in sorted_patterns]
            pattern_values = [p[1] for p in sorted_patterns]
            
            axes[1, 1].barh(pattern_names, pattern_values, color='skyblue')
            axes[1, 1].set_title('Most Common Patterns')
            axes[1, 1].set_xlabel('Frequency')
        
        # 6. System metrics
        metrics_text = []
        
        total_assessments = len(analysis_history)
        metrics_text.append(f"Total Assessments: {total_assessments}")
        
        if analysis_history:
            avg_npi = np.mean([a.get('npi_score', 0) for a in analysis_history])
            metrics_text.append(f"Average NPI: {avg_npi:.1f}")
            
            if 'classification' in analysis_history[-1]:
                last_class = analysis_history[-1]['classification']
                if isinstance(last_class, dict):
                    metrics_text.append(f"Current Level: {last_class.get('predicted_class', 'N/A')}")
                else:
                    metrics_text.append(f"Current Level: {last_class}")
        
        axes[1, 2].text(0.1, 0.5, "\n".join(metrics_text), 
                       fontsize=11, verticalalignment='center')
        axes[1, 2].set_title('System Metrics')
        axes[1, 2].axis('off')
        
        plt.suptitle('Neuro-Motor Analysis Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig