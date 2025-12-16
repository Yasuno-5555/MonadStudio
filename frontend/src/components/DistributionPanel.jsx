// Distribution Heatmap Panel (a × z)
// Visualizes wealth distribution from One-Asset HANK model
import { useRef, useEffect } from 'react';

/**
 * DistributionPanel - Canvas-based 2D heatmap
 * 
 * Props:
 *   distribution_data: Flattened array (Nz * Na), z-major order
 *   grid_a: Asset grid values (Na)
 *   grid_z: Income state values (Nz)
 *   Na: Number of asset grid points
 *   Nz: Number of income states
 */
export default function DistributionPanel({ distribution_data, grid_a, grid_z, Na, Nz }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        // Defensive: check all required data exists
        if (!distribution_data || !grid_a || !grid_z || !Na || !Nz) {
            console.warn('DistributionPanel: Missing required data', {
                distribution_data: !!distribution_data,
                grid_a: !!grid_a,
                grid_z: !!grid_z,
                Na,
                Nz
            });
            return;
        }
        if (!canvasRef.current) return;
        if (!Array.isArray(distribution_data) || distribution_data.length === 0) {
            console.warn('DistributionPanel: distribution_data is not a valid array');
            return;
        }

        try {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');

            // Canvas dimensions
            const width = canvas.width;
            const height = canvas.height;

            // Clear
            ctx.fillStyle = '#1e1e1e';
            ctx.fillRect(0, 0, width, height);

            // Find min/max for color scaling
            const maxVal = Math.max(...distribution_data);

            // Cell dimensions
            const cellWidth = (width - 60) / Na;
            const cellHeight = (height - 40) / Nz;
            const offsetX = 50;
            const offsetY = 10;

            // Draw heatmap cells
            for (let z_idx = 0; z_idx < Nz; z_idx++) {
                for (let a_idx = 0; a_idx < Na; a_idx++) {
                    const idx = z_idx * Na + a_idx;
                    const val = distribution_data[idx] || 0;

                    // Color: blue (low) -> cyan -> yellow -> red (high)
                    const intensity = maxVal > 0 ? Math.sqrt(val / maxVal) : 0;
                    const r = Math.floor(255 * Math.min(1, intensity * 2));
                    const g = Math.floor(255 * Math.min(1, intensity * 1.5) * (1 - intensity * 0.5));
                    const b = Math.floor(255 * (1 - intensity));

                    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                    ctx.fillRect(
                        offsetX + a_idx * cellWidth,
                        offsetY + (Nz - 1 - z_idx) * cellHeight,
                        cellWidth,
                        cellHeight
                    );
                }
            }

            // Draw axis labels
            ctx.fillStyle = '#8b949e';
            ctx.font = '10px monospace';

            // X-axis (Asset)
            ctx.textAlign = 'center';
            ctx.fillText('Assets (a)', width / 2, height - 5);

            // X ticks (show 5 values)
            for (let i = 0; i < 5; i++) {
                const idx = Math.floor(i * (Na - 1) / 4);
                const x = offsetX + idx * cellWidth + cellWidth / 2;
                const val = grid_a[idx];
                ctx.fillText(typeof val === 'number' ? val.toFixed(1) : '?', x, height - 18);
            }

            // Y-axis (Income)
            ctx.save();
            ctx.translate(10, height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.textAlign = 'center';
            ctx.fillText('Income (z)', 0, 0);
            ctx.restore();

            // Y ticks
            ctx.textAlign = 'right';
            for (let z_idx = 0; z_idx < Nz; z_idx++) {
                const y = offsetY + (Nz - 1 - z_idx) * cellHeight + cellHeight / 2 + 4;
                const val = grid_z[z_idx];
                ctx.fillText(`z=${typeof val === 'number' ? val.toFixed(1) : '?'}`, offsetX - 5, y);
            }

            // Color bar legend
            const legendX = width - 20;
            const legendHeight = height - 50;
            const legendY = 20;
            const legendWidth = 10;

            const gradient = ctx.createLinearGradient(0, legendY + legendHeight, 0, legendY);
            gradient.addColorStop(0, 'rgb(0, 0, 255)');
            gradient.addColorStop(0.5, 'rgb(255, 255, 0)');
            gradient.addColorStop(1, 'rgb(255, 0, 0)');
            ctx.fillStyle = gradient;
            ctx.fillRect(legendX, legendY, legendWidth, legendHeight);

            // Legend labels
            ctx.fillStyle = '#8b949e';
            ctx.textAlign = 'left';
            ctx.fillText('High', legendX + 15, legendY + 10);
            ctx.fillText('Low', legendX + 15, legendY + legendHeight);

        } catch (err) {
            console.error('DistributionPanel: Error drawing heatmap', err);
        }

    }, [distribution_data, grid_a, grid_z, Na, Nz]);

    if (!distribution_data) {
        return (
            <div style={styles.container}>
                <div style={styles.header}>Distribution</div>
                <div style={styles.empty}>Run simulation to see distribution</div>
            </div>
        );
    }

    return (
        <div style={styles.container}>
            <div style={styles.header}>Wealth Distribution (a × z)</div>
            <canvas
                ref={canvasRef}
                width={300}
                height={120}
                style={styles.canvas}
            />
            <div style={styles.note}>
                Mass sums to 1.0 (probability distribution)
            </div>
        </div>
    );
}

const styles = {
    container: {
        backgroundColor: '#161b22',
        borderRadius: '6px',
        padding: '12px',
        marginTop: '12px',
    },
    header: {
        color: '#c9d1d9',
        fontWeight: 'bold',
        marginBottom: '8px',
        fontSize: '0.9rem',
    },
    empty: {
        color: '#8b949e',
        fontSize: '0.8rem',
        fontStyle: 'italic',
    },
    canvas: {
        width: '100%',
        height: 'auto',
        borderRadius: '4px',
        border: '1px solid #30363d',
    },
    note: {
        color: '#6e7681',
        fontSize: '0.7rem',
        marginTop: '6px',
        fontStyle: 'italic',
    },
};
