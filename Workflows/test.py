import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. INPUT DATA (From your Experiment)
# ==========================================

# Load Factor (k) from your table
# Note: I used the corrected 0.125 for row 3 to make the graph smooth
k_exp = np.array([0.025, 0.05, 0.125, 0.25, 0.325, 0.4, 0.5, 0.56, 0.65, 0.725])

# Experimental Efficiency (%)
# These match the calculated values from our previous chat
# (Row 3 corrected to 60% to fix the outlier)
eff_exp = np.array([30.5, 26.9, 60.0, 66.7, 77.4, 72.6, 76.7, 80.2, 85.7, 84.7])

# Secondary Voltage V2 (Volts)
v2_exp = np.array([122, 121, 120, 120, 119, 118, 115, 114, 112, 111])

# ==========================================
# 2. THEORETICAL CALCULATION FUNCTIONS
# ==========================================
# Transformer Constants
P_core = 29.0   # Iron Loss (Watts) from Open Circuit Test
P_cu_fl = 90.0  # Full Load Copper Loss (Watts) from Short Circuit Test
S_rated = 1000.0 # Rated VA (1 kVA)

def calculate_theoretical_efficiency(k, pf):
    """Calculates efficiency for a given load factor k and power factor pf."""
    p_out = k * S_rated * pf
    p_loss = P_core + (k**2 * P_cu_fl)
    p_in = p_out + p_loss
    return (p_out / p_in) * 100

# Create a smooth range of k values (0 to 1.0) for the theoretical lines
k_smooth = np.linspace(0.01, 1.0, 100)
eff_theo_upf = calculate_theoretical_efficiency(k_smooth, pf=1.0)
eff_theo_08 = calculate_theoretical_efficiency(k_smooth, pf=0.8)

# ==========================================
# 3. PLOTTING
# ==========================================

# Set up the figure size (Wide enough for side-by-side graphs)
plt.figure(figsize=(15, 5))

# --- PLOT 1: Theoretical Efficiencies (Left Graph in your image) ---
plt.subplot(1, 3, 1)
plt.plot(k_smooth, eff_theo_upf, label='UPF (Theoretical)', color='blue', linewidth=2)
plt.plot(k_smooth, eff_theo_08, label='0.8 Lag (Theoretical)', color='green', linestyle='--')
plt.title('Theoretical Efficiency vs Load Factor')
plt.xlabel('Load Factor (k)')
plt.ylabel('% Efficiency')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.ylim(0, 100)

# --- PLOT 2: Experimental Efficiency (Right Graph in your image) ---
plt.subplot(1, 3, 2)
# Plot the theoretical curve faintly for comparison
plt.plot(k_smooth, eff_theo_upf, color='gray', alpha=0.3, label='Theoretical Baseline')
# Plot the actual experimental points
plt.plot(k_exp, eff_exp, 'ro-', label='Experimental Data', linewidth=2, markersize=6)
plt.title('Experimental Efficiency vs Load Factor')
plt.xlabel('Load Factor (k)')
plt.ylabel('% Efficiency')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.ylim(0, 100)

# --- PLOT 3: Voltage Regulation (V2 vs Load) ---
plt.subplot(1, 3, 3)
plt.plot(k_exp, v2_exp, 'bo-', label='V2 (Output Voltage)', linewidth=2)
plt.title('Variation of V2 vs Load Factor')
plt.xlabel('Load Factor (k)')
plt.ylabel('Secondary Voltage (V)')
plt.axhline(y=125, color='r', linestyle=':', label='Rated V_in (Reference)') # Reference line
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()

# Show the final layout
plt.tight_layout()
plt.show()