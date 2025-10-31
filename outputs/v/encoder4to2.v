module encoder(
    input [3:0] d,
    output [1:0] y
);

always @* begin
    case (d)
        4'b0000: y = 2'b00;
        4'b0001: y = 2'b01;
        4'b0010: y = 2'b10;
        4'b0011: y = 2'b11;
        4'b0100: y = 2'b00;
        4'b0101: y = 2'b01;
        4'b0110: y = 2'b10;
        4'b0111: y = 2'b11;
        4'b1000: y = 2'b00;
        4'b1001: y = 2'b01;
        4'b1010: y = 2'b10;
        4'b1011: y = 2'b11;
        default: y = 2'bxx; // This should never happen, but it's good practice to include a default case.
    endcase
end

endmodule
