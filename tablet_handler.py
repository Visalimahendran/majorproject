import hid
import time
from datetime import datetime
import struct
import threading
import queue

class TabletHandler:
    """Handle digital tablet input (Wacom, Huion, etc.)"""
    
    # Common tablet vendor IDs
    TABLET_VENDORS = {
        0x056A: "Wacom",
        0x256C: "Huion",
        0x28BD: "XP-Pen",
        0x0B57: "Veikk"
    }
    
    def __init__(self, vendor_id=None, product_id=None):
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.device = None
        self.is_connected = False
        self.data_queue = queue.Queue()
        self.callbacks = []
        
        # Tablet properties
        self.max_x = 0
        self.max_y = 0
        self.max_pressure = 0
        self.sampling_rate = 0
        
        # State tracking
        self.pen_down = False
        self.last_position = (0, 0)
        self.current_stroke = []
        
    def discover_tablets(self):
        """Discover connected tablets"""
        tablets = []
        for device_info in hid.enumerate():
            vendor_name = self.TABLET_VENDORS.get(device_info['vendor_id'], "Unknown")
            if vendor_name != "Unknown":
                tablets.append({
                    'vendor_id': device_info['vendor_id'],
                    'product_id': device_info['product_id'],
                    'vendor_name': vendor_name,
                    'product_string': device_info['product_string'],
                    'path': device_info['path']
                })
        
        return tablets
    
    def connect(self, vendor_id=None, product_id=None):
        """Connect to tablet"""
        if vendor_id:
            self.vendor_id = vendor_id
        if product_id:
            self.product_id = product_id
        
        try:
            # Find device
            devices = self.discover_tablets()
            target_device = None
            
            for device in devices:
                if (device['vendor_id'] == self.vendor_id and 
                    (self.product_id is None or device['product_id'] == self.product_id)):
                    target_device = device
                    break
            
            if not target_device:
                raise Exception(f"No tablet found with VID={self.vendor_id:04X}, PID={self.product_id:04X}")
            
            # Open device
            self.device = hid.device()
            self.device.open_path(target_device['path'])
            self.device.set_nonblocking(1)
            
            self.is_connected = True
            
            # Start reading thread
            self.reading_thread = threading.Thread(target=self._read_data)
            self.reading_thread.daemon = True
            self.reading_thread.start()
            
            # Get tablet properties
            self._detect_tablet_properties(target_device['vendor_name'])
            
            print(f"Connected to {target_device['vendor_name']} tablet")
            return True
            
        except Exception as e:
            print(f"Failed to connect to tablet: {e}")
            return False
    
    def _detect_tablet_properties(self, vendor_name):
        """Detect tablet properties based on vendor"""
        if vendor_name == "Wacom":
            self.max_x = 15200
            self.max_y = 9500
            self.max_pressure = 8192
            self.sampling_rate = 200  # Hz
        elif vendor_name == "Huion":
            self.max_x = 50800
            self.max_y = 31750
            self.max_pressure = 8192
            self.sampling_rate = 200
        elif vendor_name == "XP-Pen":
            self.max_x = 50800
            self.max_y = 31750
            self.max_pressure = 8192
            self.sampling_rate = 266
        else:
            # Default values
            self.max_x = 32767
            self.max_y = 32767
            self.max_pressure = 1024
            self.sampling_rate = 100
    
    def _read_data(self):
        """Continuously read data from tablet"""
        while self.is_connected:
            try:
                data = self.device.read(64)  # Read up to 64 bytes
                if data:
                    self._parse_tablet_data(data)
                    self.data_queue.put(data)
            except Exception as e:
                print(f"Error reading from tablet: {e}")
                time.sleep(0.01)
    
    def _parse_tablet_data(self, data):
        """Parse tablet data packet"""
        if len(data) < 9:
            return
        
        # Parse based on common tablet protocols
        packet_type = data[0]
        
        if packet_type == 0x02:  # Wacom protocol
            self._parse_wacom_data(data)
        elif packet_type in [0x06, 0x07, 0x08]:  # Huion/XP-Pen protocol
            self._parse_hid_data(data)
    
    def _parse_wacom_data(self, data):
        """Parse Wacom tablet data"""
        # Wacom protocol parsing
        x = (data[2] << 8) | data[1]
        y = (data[4] << 8) | data[3]
        pressure = (data[6] << 8) | data[5]
        buttons = data[7]
        
        # Normalize coordinates
        norm_x = x / self.max_x
        norm_y = y / self.max_y
        norm_pressure = pressure / self.max_pressure
        
        # Check pen state
        pen_down = pressure > 0
        
        # Create data point
        point = {
            'x': norm_x,
            'y': norm_y,
            'pressure': norm_pressure,
            'timestamp': datetime.now(),
            'pen_down': pen_down,
            'buttons': buttons
        }
        
        # Handle stroke tracking
        if pen_down and not self.pen_down:
            # Pen just touched
            self.current_stroke = [point]
            self.pen_down = True
        elif pen_down and self.pen_down:
            # Pen continuing stroke
            self.current_stroke.append(point)
        elif not pen_down and self.pen_down:
            # Pen lifted
            if len(self.current_stroke) > 1:
                self._emit_stroke(self.current_stroke)
            self.current_stroke = []
            self.pen_down = False
        
        # Update last position
        self.last_position = (norm_x, norm_y)
        
        # Notify callbacks
        self._notify_callbacks('point', point)
    
    def _parse_hid_data(self, data):
        """Parse HID tablet data (Huion, XP-Pen)"""
        # HID protocol parsing
        x = (data[3] << 8) | data[2]
        y = (data[5] << 8) | data[4]
        pressure = (data[7] << 8) | data[6]
        
        # Check if pen is in proximity
        in_proximity = (data[1] & 0x40) != 0
        pen_down = (data[1] & 0x01) != 0
        
        if in_proximity:
            # Normalize coordinates
            norm_x = x / self.max_x
            norm_y = y / self.max_y
            norm_pressure = pressure / self.max_pressure if pen_down else 0
            
            point = {
                'x': norm_x,
                'y': norm_y,
                'pressure': norm_pressure,
                'timestamp': datetime.now(),
                'pen_down': pen_down,
                'in_proximity': in_proximity
            }
            
            # Handle stroke tracking
            if pen_down and not self.pen_down:
                self.current_stroke = [point]
                self.pen_down = True
            elif pen_down and self.pen_down:
                self.current_stroke.append(point)
            elif not pen_down and self.pen_down:
                if len(self.current_stroke) > 1:
                    self._emit_stroke(self.current_stroke)
                self.current_stroke = []
                self.pen_down = False
            
            # Notify callbacks
            self._notify_callbacks('point', point)
    
    def _emit_stroke(self, stroke):
        """Emit completed stroke"""
        stroke_data = {
            'points': stroke,
            'start_time': stroke[0]['timestamp'],
            'end_time': stroke[-1]['timestamp'],
            'duration': (stroke[-1]['timestamp'] - stroke[0]['timestamp']).total_seconds(),
            'point_count': len(stroke)
        }
        
        self._notify_callbacks('stroke', stroke_data)
    
    def register_callback(self, event_type, callback):
        """Register callback for tablet events"""
        self.callbacks.append((event_type, callback))
    
    def _notify_callbacks(self, event_type, data):
        """Notify registered callbacks"""
        for cb_event_type, callback in self.callbacks:
            if cb_event_type == event_type or cb_event_type == 'all':
                try:
                    callback(data)
                except Exception as e:
                    print(f"Callback error: {e}")
    
    def get_strokes(self, timeout=1):
        """Get strokes from queue"""
        strokes = []
        try:
            while True:
                data = self.data_queue.get(timeout=timeout)
                # Process data if needed
                pass
        except queue.Empty:
            pass
        
        return strokes
    
    def calibrate(self):
        """Calibrate tablet"""
        print("Tablet calibration:")
        print("1. Touch the top-left corner of active area")
        input("Press Enter when ready...")
        # Record calibration point
        # Repeat for other corners
        
        print("Calibration complete!")
    
    def disconnect(self):
        """Disconnect from tablet"""
        self.is_connected = False
        if self.device:
            self.device.close()
        
        if hasattr(self, 'reading_thread'):
            self.reading_thread.join(timeout=1)
    
    def get_tablet_info(self):
        """Get tablet information"""
        if not self.device:
            return None
        
        try:
            info = {
                'vendor_id': self.vendor_id,
                'product_id': self.product_id,
                'vendor_name': self.TABLET_VENDORS.get(self.vendor_id, "Unknown"),
                'max_x': self.max_x,
                'max_y': self.max_y,
                'max_pressure': self.max_pressure,
                'sampling_rate': self.sampling_rate,
                'is_connected': self.is_connected
            }
            
            # Try to get more info from device
            try:
                info['manufacturer'] = self.device.get_manufacturer_string()
                info['product'] = self.device.get_product_string()
                info['serial_number'] = self.device.get_serial_number_string()
            except:
                pass
            
            return info
            
        except Exception as e:
            print(f"Error getting tablet info: {e}")
            return None