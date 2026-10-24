{
  perSystem =
    { self', ... }:
    {
      checks.pool-balls = self'.packages.pool-balls;
    };
}
