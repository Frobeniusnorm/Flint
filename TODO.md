# Developer TODOs
- Optimize Convolve gradient of second parent for small results, but large input
- Implement batch filter execution of first convolve gradient
- Allow eager gpu operations to set their own work size and cpu operations to only receive their thread id and
  share work themselfs (for caching, more optimized distribution)
