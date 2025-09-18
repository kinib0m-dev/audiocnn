export const getColor = (value: number): [number, number, number] => {
  // Ensure value is a valid number and clamp it between -1 and 1
  const clampedValue = Math.max(-1, Math.min(1, isNaN(value) ? 0 : value));

  let r: number, g: number, b: number;

  if (clampedValue > 0) {
    r = 255 * (1 - clampedValue * 0.8);
    g = 255 * (1 - clampedValue * 0.5);
    b = 255;
  } else {
    r = 255;
    g = 255 * (1 + clampedValue * 0.5);
    b = 255 * (1 + clampedValue * 0.8);
  }

  return [Math.round(r), Math.round(g), Math.round(b)];
};
