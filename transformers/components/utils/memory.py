from .helpers import generate_causal_mask

"""The Memory class is used to store relevant information that will be used by the components."""
class Memory:

    def __init__(
            self, 
            stack=None, 
            encoder=None, 
            decoder=None, 
            generate_encoder_mask=True, 
            generate_decoder_mask=True, 
            encoder_mask=None, 
            decoder_mask=None
        ):

        self.stack = stack

        self.encoder = encoder
        self.decoder = decoder

        if encoder_mask != None:
            self.encoder_mask = encoder_mask
        elif generate_encoder_mask == False or encoder == None:
            self.encoder_mask = None
        else:
            self.encoder_mask = (encoder != 0).unsqueeze(1).unsqueeze(2)

        if decoder_mask != None:
            self.decoder_mask = decoder_mask
        elif generate_decoder_mask == False or decoder == None:
            self.decoder_mask = None
        else:
            self.decoder_mask = generate_causal_mask(decoder)

        self.original_encoder = encoder
        self.original_decoder = decoder

    def get_stack(self):
        return self.stack

    def get_encoder_tensor(self):
        return self.encoder

    def get_decoder_tensor(self):
        return self.decoder

    def get_encoder_mask(self):
        return self.encoder_mask

    def get_decoder_mask(self):
        return self.decoder_mask

    def get_original_encoder_tensor(self):
        return self.original_encoder

    def get_original_decoder_tensor(self):
        return self.original_decoder

    def set_stack(self, stack):
        self.stack = stack

    def set_encoder_tensor(self, encoder):
        self.encoder = encoder

    def set_decoder_tensor(self, decoder):
        self.decoder = decoder