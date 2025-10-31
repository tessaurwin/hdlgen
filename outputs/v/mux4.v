module mux_4to1(a, b, c, d, sel[1:0], y);
  always @* begin
    case (sel)
      2'b00: y = a;
      2'b01: y = b;
      2'b10: y = c;
      2'b11: y = d;
      default: y = 0; // Optional
    endcase
  end
endmodule
