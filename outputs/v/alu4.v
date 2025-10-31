module alu(
    input [3:0] a, b, sel;
    output reg [3:0] y;
);

always @* begin
    case (sel)
        2'b00: y = a & b; // AND operation
        2'b01: y = a | b; // OR operation
        2'b10: y = a + b; // ADD operation
        2'b11: y = a ^ b; // XOR operation
    endcase
end

endmodule
