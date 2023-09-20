import argparse
import subprocess


def parse_args():
    """Parse command line arguments for the management of the Lazo Index.

    Returns:
        argparse.Namespace: Namespace containing the arguments.
    """
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(required=True)

    start = subparsers.add_parser("start", help="Start the lazo index container.")
    kill = subparsers.add_parser("kill", help="Kill the lazo index container.")
    restart = subparsers.add_parser(
        "restart", help="Restart a lazo container container."
    )
    clean = subparsers.add_parser("clean", help="Remove the given container.")
    drop = subparsers.add_parser("drop", help="Remove the given container.")

    start.add_argument(
        "--lazo_container_name",
        action="store",
        default="lazo_index",
        help="Name of the lazo container.",
    )
    start.add_argument(
        "--es_port",
        action="store",
        default=9200,
        help="Port used by Elasticsearch for listening.",
    )
    start.add_argument(
        "--lazo_port",
        action="store",
        default=15449,
        help="Port used by the lazo index. ",
    )
    start.set_defaults(function=start_command)

    kill.add_argument(
        "--container_name",
        action="store",
        default="lazo_index",
        help="Name of the container to kill.",
    )
    kill.set_defaults(function=kill_command)

    restart.add_argument(
        "--container_name",
        action="store",
        default="lazo_index",
        help="Name of the container to restart.",
    )
    restart.set_defaults(function=restart_command)

    clean.add_argument(
        "--container_name",
        action="store",
        default="lazo_index",
        help="Name of the container to remove.",
    )
    clean.add_argument(
        "--es_port",
        action="store",
        default=9200,
        help="Port used by Elasticsearch for listening (to drop the index).",
    )
    clean.set_defaults(function=clean_command)

    drop.add_argument(
        "--es_port",
        action="store",
        default=9200,
        help="Port used by Elasticsearch for listening (to drop the index).",
    )
    drop.set_defaults(function=drop_command)

    args = parser.parse_args()
    return args


def start_command(arguments):
    """Start the lazo docker container using the given parameters.

    Args:
        arguments (argparse.Namespace): Arguments to be forwarded to the docker run command.
    """
    lazo_container_name = arguments.lazo_container_name
    elasticsearc_port = arguments.es_port
    port = arguments.lazo_port

    command_str = (
        f"""docker run --name {lazo_container_name} -e DATABASE=elasticsearch -e """
        f"""ELASTICSEARCH_PORT={elasticsearc_port} -e PORT={port}"""
        f""" --network="host" registry.gitlab.com/vida-nyu/auctus/lazo-index-service:0.7.2 &"""
    )
    print(command_str)

    subprocess.run(command_str, shell=True, check=True)


def kill_command(arguments: argparse.Namespace):
    """Kill the container with the given argument.

    Args:
        arguments (argparse.Namespace): Arguments to be forwarded to the docker kill command.
    """
    command_str = f"docker kill {arguments.container_name}"

    subprocess.run(command_str, shell=True, check=True)


def restart_command(arguments: argparse.Namespace):
    """Restart the docker container with the given name.

    Args:
        arguments (argparse.Namespace): Arguments to be forwarded to the docker restart command.
    """
    command_str = f"docker restart {arguments.container_name}"

    subprocess.run(command_str, shell=True, check=True)


def clean_command(arguments: argparse.Namespace):
    """Clean the Elasticsearch database from the lazo index and remove the docker container.

    Args:
        arguments (argparse.Namespace): Arguments to be forwarded to the commands.

    Raises:
        RuntimeError: Raise RuntimeError if the container could not be removed.
    """
    command_str = f"docker rm {arguments.container_name}"
    try:
        subprocess.run(command_str, shell=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("The container could not be removed.") from exc

    command_str = f'curl -X DELETE "localhost:{arguments.es_port}/lazo?pretty"'
    subprocess.run(command_str, shell=True, check=True)


def drop_command(arguments: argparse.Namespace):
    """Drop the lazo index from Elasticsearch without removing the lazo docker container.

    Args:
        arguments (argparse.Namespace): Arguments to be forwarded to the commands.
    """

    command_str = f'curl -X DELETE "localhost:{arguments.es_port}/lazo?pretty"'
    subprocess.run(command_str, shell=True, check=True)


if __name__ == "__main__":
    args = parse_args()
    # start_command(args)
    print(args)
    args.function(args)
