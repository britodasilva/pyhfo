import sys, struct, math
import numpy as np
"""
Got in https://github.com/ntopper/RHD.py.git
Wrapper for rhd file fromat
See http://www.intantech.com/files/Intan_RHD2000_data_file_formats.pdf for details
"""

class RHD:
    def __init__(self, rhd_file):
        """
        Constructor takes in open file object
        you can use use RHD.openRhd(file_path) 
        to follow python convention
        """
        
        #list of enabled channel names
        self._AMPLIFIER_CHANNELS = []
        self._AUX_CHANNELS = []
        self._SUPPLY_VOLTAGE_CHANNELS = []
        self._ADC_INPUT_CHANNELS = []
        self._DIGITAL_INPUT_CHANNELS = []
        
        #number of temp sensors
        self._TEMP_SENSORS = 0
        
        self.rhd = rhd_file
        self.readHead()
        self.readBlocks()

    def readHead(self):
        """
        Reads all header data from the rhd file
        creates signal group and channel list
        """
        filesize = self.rhd.tell()
        
        #the order in which all of this is called is critcal
        self.header_identifier = hex(np.uint32(struct.unpack('<I', self.rhd.read(4))))
        v = np.int8(struct.unpack('BBBB', self.rhd.read(4)))

        #read each property of the header
        self.version = str(v[0]) + '.' + str(v[2])
        self.sample_rate = np.float32(struct.unpack('f', self.rhd.read(4)))[0] 
        self.dsp_enabled = np.int8(struct.unpack('BB', self.rhd.read(2)))[0]
        self.actual_dsp_cutoff_frequency = np.float32(struct.unpack('f', self.rhd.read(4)))[0]
        self.actual_lower_bandwidth = np.float32(struct.unpack('f', self.rhd.read(4)))[0]
        self.actual_upper_bandwidth = np.float32(struct.unpack('f', self.rhd.read(4)))[0]
        self.desired_dsp_cutoff_frequency = np.float32(struct.unpack('f', self.rhd.read(4)))[0]
        self.desired_lower_bandwidth = np.float32(struct.unpack('f', self.rhd.read(4)))[0]
        self.desired_upper_bandwidth = np.float32(struct.unpack('f', self.rhd.read(4)))[0]
        self.notch_cutoff_mode = np.int8(struct.unpack('BB', self.rhd.read(2)))[0]
        self.desired_impedance_test_frequency = np.float32(struct.unpack('f', self.rhd.read(4)))[0]
        self.actual_impedance_test_frequency = np.float32(struct.unpack('f', self.rhd.read(4)))[0]
        #list of 3 notes
        self.note = [_qstring(self.rhd),_qstring(self.rhd),_qstring(self.rhd)]
        self.number_of_temperature_sensors = np.int16(struct.unpack('h', self.rhd.read(2)))[0]
        self._TEMP_SENSORS = self.number_of_temperature_sensors
        self.board_mode = np.int16(struct.unpack('h', self.rhd.read(2)))[0]
        self.number_of_signal_groups =  np.int16(struct.unpack('h', self.rhd.read(2)))[0]

        #dict of signal groups
        self.signal_groups = {}     
        for i in range(self.number_of_signal_groups):
            sg = Signal_Group(self)
            self.signal_groups[sg.signal_group_name] = sg
        
        #dict of channels
        self.channels = {}
        for key, group in self.signal_groups.iteritems():
            self.channels.update(group.channels)

    def readBlocks(self):
        """
        Reads data blocks untill the end of file is reached
        appends each block to instance data_block_list
        """
        self.data_block_list = []
        self.data_block_list.append(Rhd2000DataBlock(self))
        #read data blocks untill the EOF
        while True:
            try:
                self.data_block_list.append(Rhd2000DataBlock(self))
            except:
                break

class Rhd2000DataBlock():
        """
        Holds all data stored in a single data block of an RHD file 
        """
        
        def __init__(self, parent):
            """
            takes in parent RHD instance
            Reads the next block of data in the parent's file object
            parses all data from block and stoes in insance
            """
            
            #60 32 bit integers are recorded for the amplifier sample time index        
            self.sample_time_index = []
            for i in range(60):
                sample_time = np.int32(struct.unpack('i', parent.rhd.read(4)))[0]
                self.sample_time_index.append(sample_time)

            #Amplifier voltages for each channel
            self.electrode_traces = {}#key: channel name value: voltage trce
            for amp in parent._AMPLIFIER_CHANNELS:
                electrode_voltage_trace = []
                #60 samples per channel, int16
                for i in range(60):
                    electrode_voltage = np.uint16(struct.unpack('H', parent.rhd.read(2)))[0]
                    electrode_voltage_trace.append(electrode_voltage)
                self.electrode_traces[amp] = electrode_voltage_trace    

            #Get voltage from Aux input channels
            self.auxilary_traces = {}
            for aux in parent._AUX_CHANNELS:
                aux_voltage_trace = []
                #15 samples per channel, int16
                for i in range(15):
                    aux_voltage = np.uint16(struct.unpack('H', parent.rhd.read(2)))[0]
                    aux_voltage_trace.append(aux_voltage)
                self.auxilary_traces[aux] = aux_voltage_trace   

            #get voltage from supply voltage channels
            self.supply_voltages = {}
            for sup in parent._SUPPLY_VOLTAGE_CHANNELS:
                sup_voltage_list = []
                for i in range(1):
                    sup_voltage = np.uint16(struct.unpack('H', parent.rhd.read(2)))[0]
                    sup_voltage_list.append(sup_voltage)
                self.supply_voltages[sup] = sup_voltage_list    

            #get voltage from temerature sensor channels
            self.temerature_sensor_readings = {}
            for n in range(parent._TEMP_SENSORS):
                temp_list = []
                for i in range(1):
                    temperature = np.int16(struct.unpack('h', parent.rhd.read(2)))[0]
                    temp_list.append(temperature)
                self.temerature_sensor_readings[n] = temp_list 

            #Get voltage ADC inputs
            self.board_adc_input_voltages = {}
            for adc in parent._ADC_INPUT_CHANNELS:
                adc_input_list = []
                for i in range(60):
                    adc_input = np.uint16(struct.unpack('H', parent.rhd.read(2)))[0]
                    adc_input_list.append(adc_input)
                self.board_adc_input_voltages[adc] = adc_input_list   

            #Get digital input values
            self.board_digital_inputs = {}
            for dig in parent._DIGITAL_INPUT_CHANNELS :
                digital_input_list = []
                for i in range(60):
                    digital_input = np.uint16(struct.unpack('H', parent.rhd.read(2)))[0]
                    digital_input_list.append(digital_input)
                self.board_digital_inputs[dig.native_channel_name] = digital_input_list
        
        def getTrace(self, signal_type, native_channel_name):
            """
            given a signal type and a channel name,
            returns the raw trace of data from this block for the given channel
            """
            
            dicts = [self.electrode_traces,
                     self.auxilary_traces,
                     self.supply_voltages,
                     self.board_adc_input_voltages,
                     self.board_digital_inputs,
                     self.temerature_sensor_readings]
            dictionary = dicts[signal_type]

            return dictionary[native_channel_name]

class Signal_Group():
    """
    Represents a single "Signal Group" and stores all revelent data
    """
    def __init__(self, parent):     
        """
        Reads in and parses the next chunk of signal group data the parent RHD instance
        creates a dict of chanel objects for the given signal group
        """   
        self.signal_group_name = _qstring(parent.rhd)
        self.signal_group_header = _qstring(parent.rhd)
        self.signal_group_enabled = (np.int16(struct.unpack('h', parent.rhd.read(2)))[0] == 1)
        self.number_of_channels = np.int16(struct.unpack('h', parent.rhd.read(2)))[0]
        self.number_of_amplifier_channels = np.int16(struct.unpack('h', parent.rhd.read(2)))[0]
        
        self.channels = {}
        #if there are channels:
        if self.signal_group_enabled and self.number_of_channels != 0: 
            for i in range(self.number_of_channels):
                c = Channel(parent)
                self.channels[c.native_channel_name] = c

#contains channel metadata
#reads from a .rdh file object, must be called at the correct time
class Channel():
    """
    Represents a single "Channel" and stores all revelent data
    """
    def __init__(self, parent): 
            """
            Reads in and parses the next chunk of channel group data from the parent RHD instance
            creates a dict of chanel objects for the given signal group
            """ 
            
            self.parent = parent
            
            self.custom_channel_name = _qstring(parent.rhd)
            self.native_channel_name = _qstring(parent.rhd)
            self.native_order = np.int16(struct.unpack('h', parent.rhd.read(2)))[0]
            self.custom_order =  np.int16(struct.unpack('h', parent.rhd.read(2)))[0]
            self.signal_type  = np.int16(struct.unpack('h', parent.rhd.read(2)))[0]
            self.channel_enabled = np.int16(struct.unpack('h', parent.rhd.read(2)))[0]
            self.chip_channel = np.int16(struct.unpack('h', parent.rhd.read(2)))[0]
            self.board_stream = np.int16(struct.unpack('h', parent.rhd.read(2)))[0]
            self.spike_scope_voltage_trigger_mode= np.int16(struct.unpack('h', parent.rhd.read(2)))[0]
            self.spike_scope_voltage_threshold = np.int16(struct.unpack('h', parent.rhd.read(2)))[0]
            self.spike_scope_digital_trigger_channel =  np.int16(struct.unpack('h', parent.rhd.read(2)))[0]
            self.spike_scope_digital_edge_polarity = np.int16(struct.unpack('h', parent.rhd.read(2)))[0]
            self.electrode_impedance_magnitude =  np.float32(struct.unpack('f', parent.rhd.read(4)))[0]
            self.electrode_impedance_phase = np.float32(struct.unpack('f', parent.rhd.read(4)))[0]

            if self.signal_type == 0 and self.channel_enabled:#Add name to the amplifier channel list
                parent._AMPLIFIER_CHANNELS.append(self.native_channel_name)

            if self.signal_type == 1 and self.channel_enabled:#Add name to the aux channel list
                parent._AUX_CHANNELS.append(self.native_channel_name)

            if self.signal_type == 2 and self.channel_enabled:#Supply voltage
                parent._SUPPLY_VOLTAGE_CHANNELS.append(self.native_channel_name)

            if self.signal_type == 3 and self.channel_enabled:#usb board adc input channel
                parent._ADC_INPUT_CHANNELS.append(self.native_channel_name)

            if self.signal_type == 4 and self.channel_enabled:#usb board digital input channel
                parent._DIGITAL_INPUT_CHANNELS.append(self.native_channel_name)

    def getTrace(self):
        """
        returns the raw trace of all data samples from this channel
        """
        trace = np.array([])
        for block in self.parent.data_block_list:
            
            trace = np.append(trace, block.getTrace(self.signal_type, self.native_channel_name))
        return trace
 
def openRhd(file_path):
    """
    returns a RHD object containing data from the file at the given filepath
    """
    f = open(file_path)
    return RHD(f)
    f.close()

def _qstring(f):
    """
    reads the next qstring in a file, returns a string, or null
    """
    
    length_header = np.uint32(struct.unpack('I', f.read(4)))[0]
    if length_header == int('ffffffff' , 16): #ffffffff specifies a null string
        return
    if length_header == 0: #0 is an empty string
        return ''
    string = f.read(length_header)
    
    #decoding hack, enables dictionary call by ascii (rather than hex)
    ascii = "".join(list(string)[::2])
    
    return ascii
