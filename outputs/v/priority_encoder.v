module priority_encoder(
    input  wire [7:0] d,
    output wire [2:0] y
);

always @* begin
    if (d[7]) begin
        y = 3'b100; // highest priority
    end else if (d[6]) begin
        y = 3'b010;
    end else if (d[5]) begin
        y = 3'b001;
    end else if (d[4]) begin
        y = 3'b110; // lowest priority
    end else if (d[3]) begin
        y = 3'b101;
    end else if (d[2]) begin
        y = 3'b011;
    end else if (d[1]) begin
        y = 3'b000; // tie, all inputs are low
    end else begin
        y = 3'b111; // tie, all inputs are high
    end
end

endmodule
