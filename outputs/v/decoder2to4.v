module Decoder_2_4(a, en, y);
  input wire [1:0] a;
  input wire en;
  output wire [3:0] y;
  
  always @(*) begin
    if (en) begin
      case (a)
        2'b00: y = 4'b0000;
        2'b01: y = 4'b0001;
        2'b10: y = 4'b0010;
        2'b11: y = 4'b0011;
      endcase
    end else begin
      y = 4'bzzzz;
    end
  end
endmodule
