# Measurement Implementation - Based on spectrometer_characterization.py

## ✅ Fixed Issues

### 1. **Error Code -5 Fixed**
- **Problem**: "Function is called while result of previous function is not received yet"
- **Solution**: Added proper delays and sequential operations exactly like characterization script
- **Key Changes**:
  - Added `time.sleep(0.2)` after `spec.set_it()`
  - Added `time.sleep(0.5)` before measurements
  - Added `time.sleep(2)` after turning off lasers
  - Sequential laser control (turn off all, then turn on specific)

### 2. **Measurement Tab Enhanced**
- **Live Plot**: Added matplotlib plot exactly like characterization script
  - Real-time spectrum display during auto-IT adjustment
  - Shows current laser, integration time, and peak value
  - Same layout and styling as characterization script
- **Laser Selection**: Updated to match characterization script
  - Default lasers: ["532", "445", "405", "377", "Hg_Ar"]
  - All selected by default
  - Same order as characterization script

### 3. **Exact Workflow Implementation**
The measurement sequence now follows `spectrometer_characterization.py` exactly:

```python
# 1. Turn off all lasers initially
for ch in range(1, 6): obis_laser_off(ch)
cube_laser_off()
relay_off(1); relay_off(2); relay_off(3)

# 2. For each selected laser:
for lwl in selected_lasers:
    # Turn on specific laser with exact same logic
    if lwl == "377": cube_laser_on(12)
    elif lwl == "517": relay_on(3)
    elif lwl == "532": relay_on(1)
    elif lwl == "Hg_Ar": 
        # Show fiber switch dialog
        relay_on(2)
    else: # OBIS lasers
        obis_laser_on(channel)
        set_obis_power(channel, power)
    
    # Auto-adjust integration time with live plotting
    # Take signal measurement (50 cycles)
    # Turn off laser
    # Take dark measurement (50 cycles)
```

## 🎯 Key Features Implemented

### **Auto-Integration Time Adjustment**
- Target range: 60,000 - 65,000 counts (same as characterization script)
- Anti-saturation guard at 65,400 counts
- Proportional adjustment algorithm
- Live plotting during adjustment
- Maximum 1000 iterations safety limit

### **Laser Control**
- **OBIS Lasers** (405, 445, 488 nm): Channel-based control with power setting
- **CUBE Laser** (377 nm): Power control in mW
- **Relay Lasers** (517, 532 nm, Hg-Ar): On/off control
- **Fiber Switch Dialog**: For Hg-Ar lamp (like characterization script)

### **Data Management**
- Same CSV format as characterization script
- Timestamp-based filenames
- Signal and dark measurements
- Automatic folder creation: `data/[spectrometer_sn]/`

### **Live Plotting**
- Real-time spectrum display during measurements
- Title shows: "Live Measurement for {laser} nm | IT = {time} ms | peak={counts}"
- Same axis labels and grid as characterization script
- Updates during auto-IT adjustment

## 🚀 How to Use

### **Measurements Tab**
1. **Select Lasers**: Check desired lasers (default: all selected)
2. **Set Auto-IT**: Optional starting integration time
3. **Click "Run Selected"**: Starts automated sequence
4. **Monitor Progress**: Live plot shows real-time data
5. **Results**: Data automatically saved, plots generated, shown in Analysis tab

### **Live View Tab**
- **Laser Controls**: Individual laser on/off checkboxes
- **Integration Time**: Manual IT control with "Apply IT" button
- **Live Plotting**: Real-time spectrum display

### **Setup Tab**
- **COM Ports**: Configure OBIS, CUBE, RELAY ports
- **Laser Powers**: Set power levels for each laser
- **Spectrometer**: Connect/disconnect and DLL path

## 🔧 Technical Details

### **Error Handling**
- Proper delays prevent error code -5
- Graceful handling of missing hardware
- Automatic laser shutdown on errors
- Progress reporting and error messages

### **Threading**
- Measurement runs in background thread
- UI updates via `self.after()` calls
- Thread-safe plot updates
- Proper cleanup on stop/error

### **Hardware Compatibility**
- Same COM port assignments as characterization script
- Same laser power settings
- Same measurement parameters (N_SIG=50, N_DARK=50)
- Compatible with existing hardware setup

## ✅ Verification

All functionality tested with mock hardware:
- ✅ Laser control commands sent correctly
- ✅ Auto-IT adjustment algorithm works
- ✅ Live plotting updates properly
- ✅ Data saving and analysis generation
- ✅ Error handling and cleanup
- ✅ UI responsiveness maintained

The implementation now exactly matches `spectrometer_characterization.py` workflow but integrated into the GUI!
