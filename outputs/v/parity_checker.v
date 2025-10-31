module parity_checker (
    input  [7:0] data,
    output reg parity
);

always @* begin
    parity = |data; // compute the bitwise OR of all the bits in data
end

endmodule
