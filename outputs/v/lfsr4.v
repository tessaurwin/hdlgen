module lfsr(
    input clk,
    output reg [3:0] q
);

// Taps for the LFSR
localparam tap[3] = {1'b1, 1'b1, 1'b0};

// Internal register to store the state of the LFSR
reg [3:0] s;

always @(posedge clk) begin
    // Update the internal state of the LFSR
    s <= {s[2], s[1], s[0]};
    
    // XOR the output with the taps to create a new output value
    q <= s ^ tap;
end
