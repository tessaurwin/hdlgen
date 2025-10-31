module pwm_gen (
    input clk,
    input rst_n,
    input[7:0] duty,
    output pwm_out
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // Initialize the counter and set the PWM output to 0
        cnt <= 8'h0;
        pwm_out <= 1'b0;
    end else begin
        // Increment the counter
        cnt <= cnt + 1'b1;

        // Check if the counter has reached the desired duty cycle
        if (cnt == duty) begin
            // Set the PWM output to 1
            pwm_out <= 1'b1;
        end else begin
            // Keep the PWM output at 0
            pwm_out <= 1'b0;
        end
    end
end

// Counter for tracking the number of clock cycles
reg [7:0] cnt;
